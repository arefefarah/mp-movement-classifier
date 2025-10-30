"""
Multi-Motion Analysis for finding best cutoff frequency in motion capture data smoothing
Analyzes all motions for a single subject and creates summary table
"""

import numpy as np
import re
import os
import matplotlib
import json
import pandas as pd
from glob import glob

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, filtfilt, welch, find_peaks
import warnings


def parse_bvh_robust(file_path):
    """Robust BVH parser that handles various format issues"""
    with open(file_path, 'r') as file:
        content = file.read()

    parts = content.split('MOTION')
    if len(parts) < 2:
        print("❌ Invalid BVH format: No MOTION section found")
        return None, None, None, None

    hierarchy = parts[0]
    motion_part = parts[1]

    joints = {}
    channel_index = 0
    joint_pattern = r'(ROOT|JOINT)\s+(\w+)'
    channel_pattern = r'CHANNELS\s+(\d+)\s+(.*)'

    lines = hierarchy.split('\n')
    current_joint = None

    for line in lines:
        line = line.strip()
        joint_match = re.search(joint_pattern, line)
        if joint_match:
            current_joint = joint_match.group(2)

        channel_match = re.search(channel_pattern, line)
        if channel_match and current_joint:
            num_channels = int(channel_match.group(1))
            channels = channel_match.group(2).split()
            joints[current_joint] = {
                'channels': channels,
                'start_index': channel_index
            }
            channel_index += num_channels

    motion_lines = motion_part.strip().split('\n')
    frames = 0
    frame_time = 0.0

    for line in motion_lines:
        if line.startswith('Frames:'):
            frames = int(line.split(':')[1].strip())
        elif line.startswith('Frame Time:'):
            frame_time = float(line.split(':')[1].strip())

    if frames == 0:
        print("❌ No frame information found in BVH file")
        return None, None, None, None

    frame_data = []
    for line in motion_lines:
        line = line.strip()
        if line and not line.startswith('Frames') and not line.startswith('Frame Time'):
            try:
                values = [float(x) for x in line.split()]
                frame_data.extend(values)
            except ValueError:
                continue

    total_channels = sum(len(joint['channels']) for joint in joints.values())
    expected_data_points = total_channels * frames

    if len(frame_data) < expected_data_points:
        available_frames = len(frame_data) // total_channels
        motion_data = np.array(frame_data[:available_frames * total_channels]).reshape(available_frames, total_channels)
        frames = available_frames
    else:
        motion_data = np.array(frame_data[:expected_data_points]).reshape(frames, total_channels)

    return joints, motion_data, frame_time, frames


def comprehensive_cutoff_analysis(motion_data, joints, sampling_rate=30,
                                  cutoff_range=None, filter_order=4, verbose=True):
    """Comprehensive analysis to determine optimal cutoff frequency"""
    if cutoff_range is None:
        cutoff_range = np.arange(3, 11, 0.5)

    nyquist_freq = sampling_rate / 2.0
    n_frames, n_channels = motion_data.shape

    results = {
        'cutoff_freqs': [],
        'signal_to_noise_ratio': [],
        'variance_retained': [],
        'power_spectrum_95_percentile': [],
        'smoothness_score': [],
        'rmse_vs_original': []
    }

    if verbose:
        print(f"\nData: {n_frames} frames, {n_channels} channels")

    original_variance = np.var(motion_data)

    # Compute power spectrum
    channel_power_cutoffs = []
    for ch in range(n_channels):
        freqs, psd = welch(motion_data[:, ch], fs=sampling_rate)
        cumulative_power = np.cumsum(psd) / np.sum(psd)
        freq_95_power = freqs[np.argmax(cumulative_power >= 0.95)]
        channel_power_cutoffs.append(freq_95_power)
    avg_power_cutoff = np.mean(channel_power_cutoffs)

    if verbose:
        print(f"95% power frequency: {avg_power_cutoff:.2f} Hz")

    for cutoff in cutoff_range:
        if cutoff >= nyquist_freq:
            continue

        results['cutoff_freqs'].append(cutoff)

        normalized_cutoff = cutoff / nyquist_freq
        b, a = butter(filter_order, normalized_cutoff, btype='low')

        filtered_data = np.zeros_like(motion_data)
        for ch in range(n_channels):
            filtered_data[:, ch] = filtfilt(b, a, motion_data[:, ch])

        # Metrics
        filtered_variance = np.var(filtered_data)
        variance_retained = (filtered_variance / original_variance) * 100
        results['variance_retained'].append(variance_retained)

        rmse = np.sqrt(np.mean((motion_data - filtered_data) ** 2))
        results['rmse_vs_original'].append(rmse)

        noise_estimate = motion_data - filtered_data
        signal_power = np.mean(filtered_data ** 2)
        noise_power = np.mean(noise_estimate ** 2)
        snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
        results['signal_to_noise_ratio'].append(snr)

        smoothness = np.mean([np.mean(np.abs(np.gradient(np.gradient(filtered_data[:, ch]))))
                              for ch in range(n_channels)])
        results['smoothness_score'].append(smoothness)

        results['power_spectrum_95_percentile'].append(avg_power_cutoff)

    # Convert to arrays
    for key in results.keys():
        results[key] = np.array(results[key])

    # Normalize metrics
    normalized_metrics = {}
    for key in ['variance_retained', 'signal_to_noise_ratio']:
        values = results[key]
        normalized_metrics[key] = (values - values.min()) / (values.max() - values.min() + 1e-10)

    for key in ['rmse_vs_original', 'smoothness_score']:
        values = results[key]
        normalized_metrics[key] = 1 - (values - values.min()) / (values.max() - values.min() + 1e-10)

    # Composite score
    weights = {
        'variance_retained': 0.25,
        'signal_to_noise_ratio': 0.25,
        'rmse_vs_original': 0.25,
        'smoothness_score': 0.25
    }

    composite_score = sum(normalized_metrics[key] * weights[key] for key in weights.keys())
    results['composite_score'] = composite_score
    results['normalized_metrics'] = normalized_metrics

    # Find optimal
    optimal_idx = np.argmax(composite_score)
    optimal_cutoff = results['cutoff_freqs'][optimal_idx]

    recommendations = {
        'optimal_cutoff': optimal_cutoff,
        'composite_score_at_optimal': composite_score[optimal_idx],
        'power_spectrum_95_percentile': avg_power_cutoff,
        'suggested_range': [
            max(3.0, optimal_cutoff - 1.5),
            min(nyquist_freq * 0.8, optimal_cutoff + 1.5)
        ],
        'variance_retained_at_optimal': results['variance_retained'][optimal_idx],
        'snr_at_optimal': results['signal_to_noise_ratio'][optimal_idx]
    }

    results['recommendations'] = recommendations
    return results


def plot_comprehensive_analysis(results, motion_name, save_dir):
    """Create comprehensive visualization"""
    os.makedirs(save_dir, exist_ok=True)

    cutoffs = results['cutoff_freqs']

    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(f'Cutoff Analysis: {motion_name}', fontsize=16, fontweight='bold')

    # Variance Retained
    ax = axes[0, 0]
    ax.plot(cutoffs, results['variance_retained'], 'b-o', linewidth=2, markersize=4)
    ax.axhline(y=95, color='r', linestyle='--', alpha=0.5, label='95% threshold')
    ax.set_xlabel('Cutoff Frequency (Hz)', fontsize=10)
    ax.set_ylabel('Variance Retained (%)', fontsize=10)
    ax.set_title('Signal Variance Retained', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # RMSE
    ax = axes[0, 1]
    ax.plot(cutoffs, results['rmse_vs_original'], 'g-o', linewidth=2, markersize=4)
    ax.set_xlabel('Cutoff Frequency (Hz)', fontsize=10)
    ax.set_ylabel('RMSE (degrees)', fontsize=10)
    ax.set_title('RMSE vs Original', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # SNR
    ax = axes[1, 0]
    ax.plot(cutoffs, results['signal_to_noise_ratio'], 'm-o', linewidth=2, markersize=4)
    ax.set_xlabel('Cutoff Frequency (Hz)', fontsize=10)
    ax.set_ylabel('SNR (dB)', fontsize=10)
    ax.set_title('Signal-to-Noise Ratio', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Smoothness
    ax = axes[1, 1]
    ax.plot(cutoffs, results['smoothness_score'], 'c-o', linewidth=2, markersize=4)
    ax.set_xlabel('Cutoff Frequency (Hz)', fontsize=10)
    ax.set_ylabel('Smoothness Score', fontsize=10)
    ax.set_title('Movement Smoothness', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Power Spectrum
    ax = axes[2, 0]
    avg_95_power = results['power_spectrum_95_percentile'][0]
    ax.axhline(y=avg_95_power, color='r', linestyle='--', linewidth=2,
               label=f'95% Power at {avg_95_power:.1f} Hz')
    ax.set_xlabel('Cutoff Frequency (Hz)', fontsize=10)
    ax.set_ylabel('Frequency (Hz)', fontsize=10)
    ax.set_title('Power Spectrum Reference', fontweight='bold')
    ax.set_ylim([0, 15])
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Composite Score
    ax = axes[2, 1]
    composite = results['composite_score']
    ax.plot(cutoffs, composite, 'k-o', linewidth=2, markersize=6)

    optimal_idx = np.argmax(composite)
    optimal_cutoff = cutoffs[optimal_idx]
    ax.axvline(x=optimal_cutoff, color='r', linestyle='--', linewidth=2,
               label=f'Optimal: {optimal_cutoff:.1f} Hz')
    ax.scatter([optimal_cutoff], [composite[optimal_idx]],
               color='r', s=200, marker='*', zorder=5)

    ax.set_xlabel('Cutoff Frequency (Hz)', fontsize=10)
    ax.set_ylabel('Composite Score (0-1)', fontsize=10)
    ax.set_title('Composite Score', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{motion_name}_analysis.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


def get_motion_numbers_for_subject(subject_id, bvh_dir):
    """
    Find all motion numbers available for a given subject

    """
    pattern = os.path.join(bvh_dir, f"subject_{subject_id}_motion_*.bvh")
    files = glob(pattern)

    motion_numbers = []
    for file in files:
        # Extract motion number from filename
        basename = os.path.basename(file)
        match = re.search(r'motion_(\d+)', basename)
        if match:
            motion_numbers.append(int(match.group(1)))

    return sorted(motion_numbers)


def analyze_all_motions(subject_id, bvh_dir, output_dir, motion_mapping_file):
    """
    Analyze all motions for a single subject

    Returns:
        DataFrame with summary results
    """
    # Load motion mapping
    with open(motion_mapping_file) as f:
        reverse_map = {v: k for k, v in json.load(f)['mapping'].items()}

    # Get all motion numbers for this subject
    motion_numbers = get_motion_numbers_for_subject(subject_id, bvh_dir)

    if not motion_numbers:
        print(f"❌ No motion files found for subject {subject_id}")
        return None

    print(f"\n{'=' * 80}")
    print(f"ANALYZING SUBJECT {subject_id} - {len(motion_numbers)} MOTIONS FOUND")
    print(f"{'=' * 80}\n")
    print(f"Motion numbers: {motion_numbers}")

    # Create subject-specific output directory
    subject_output_dir = os.path.join(output_dir, f"subject_{subject_id}")
    os.makedirs(subject_output_dir, exist_ok=True)

    # Store results
    results_summary = []

    # Analyze each motion
    for i, motion_num in enumerate(motion_numbers, 1):
        print(f"\n{'-' * 80}")
        print(f"[{i}/{len(motion_numbers)}] Processing Motion {motion_num}")
        print(f"{'-' * 80}")

        # Get motion name
        motion_name = reverse_map.get(motion_num, f"unknown_{motion_num}")
        bvh_filename = f"subject_{subject_id}_motion_{motion_num:02d}"
        bvh_file = os.path.join(bvh_dir, f"{bvh_filename}.bvh")

        # Check if file exists
        if not os.path.exists(bvh_file):
            print(f"⚠️  File not found: {bvh_file}")
            continue

        print(f"Motion name: {motion_name}")
        print(f"File: {bvh_filename}.bvh")

        # Parse BVH
        joints, motion_data, frame_time, frames = parse_bvh_robust(bvh_file)

        if joints is None:
            print(f"❌ Failed to parse {bvh_filename}")
            continue

        print(f"✅ Loaded: {frames} frames, {len(joints)} joints")

        # Run analysis
        try:
            results = comprehensive_cutoff_analysis(
                motion_data,
                joints,
                sampling_rate=30,
                cutoff_range=np.arange(3, 11, 0.5),
                filter_order=4,
                verbose=False
            )

            # Create motion-specific directory
            motion_output_dir = os.path.join(subject_output_dir, motion_name)
            os.makedirs(motion_output_dir, exist_ok=True)

            # Save plot
            plot_comprehensive_analysis(results, motion_name, motion_output_dir)

            # Extract key metrics
            rec = results['recommendations']
            results_summary.append({
                'Motion_Number': motion_num,
                'Motion_Name': motion_name,
                'Optimal_Cutoff_Hz': rec['optimal_cutoff'],
                'Composite_Score': rec['composite_score_at_optimal'],
                'Variance_Retained_%': rec['variance_retained_at_optimal'],
                'SNR_dB': rec['snr_at_optimal'],
                'Power_95%_Hz': rec['power_spectrum_95_percentile'],
                'Frames': frames,
                'Channels': motion_data.shape[1]
            })

            print(f"✅ Optimal cutoff: {rec['optimal_cutoff']:.1f} Hz")
            print(f"   Variance retained: {rec['variance_retained_at_optimal']:.1f}%")
            print(f"   Composite score: {rec['composite_score_at_optimal']:.3f}")

        except Exception as e:
            print(f"❌ Error analyzing {motion_name}: {str(e)}")
            continue

    # Create summary DataFrame
    if not results_summary:
        print("\n❌ No results to summarize")
        return None

    df_summary = pd.DataFrame(results_summary)

    # Sort by motion number
    df_summary = df_summary.sort_values('Motion_Number').reset_index(drop=True)

    # Save summary table
    summary_file = os.path.join(subject_output_dir, f"subject_{subject_id}_cutoff_summary.csv")
    df_summary.to_csv(summary_file, index=False, float_format='%.2f')
    print(f"\n✅ Summary table saved: {summary_file}")

    # Create visual summary table
    create_summary_plot(df_summary, subject_id, subject_output_dir)

    return df_summary


def create_summary_plot(df_summary, subject_id, output_dir):
    """Create visual summary table and plots"""

    # Summary statistics table plot
    fig, ax = plt.subplots(figsize=(14, len(df_summary) * 0.4 + 2))
    ax.axis('tight')
    ax.axis('off')

    # Prepare table data
    table_data = []
    table_data.append(['Motion #', 'Motion Name', 'Optimal\nCutoff (Hz)',
                       'Score', 'Var %', 'SNR (dB)', 'Power 95%'])

    for _, row in df_summary.iterrows():
        table_data.append([
            f"{row['Motion_Number']:02d}",
            row['Motion_Name'],
            f"{row['Optimal_Cutoff_Hz']:.1f}",
            f"{row['Composite_Score']:.3f}",
            f"{row['Variance_Retained_%']:.1f}",
            f"{row['SNR_dB']:.1f}",
            f"{row['Power_95%_Hz']:.1f}"
        ])

    # Add summary row
    table_data.append(['', '', '', '', '', '', ''])
    table_data.append([
        'MEAN',
        '',
        f"{df_summary['Optimal_Cutoff_Hz'].mean():.1f}",
        f"{df_summary['Composite_Score'].mean():.3f}",
        f"{df_summary['Variance_Retained_%'].mean():.1f}",
        f"{df_summary['SNR_dB'].mean():.1f}",
        f"{df_summary['Power_95%_Hz'].mean():.1f}"
    ])
    table_data.append([
        'STD',
        '',
        f"{df_summary['Optimal_Cutoff_Hz'].std():.1f}",
        f"{df_summary['Composite_Score'].std():.3f}",
        f"{df_summary['Variance_Retained_%'].std():.1f}",
        f"{df_summary['SNR_dB'].std():.1f}",
        f"{df_summary['Power_95%_Hz'].std():.1f}"
    ])

    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.08, 0.25, 0.12, 0.10, 0.10, 0.12, 0.12])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)

    # Style header
    for j in range(7):
        table[(0, j)].set_facecolor('#4CAF50')
        table[(0, j)].set_text_props(weight='bold', color='white')

    # Style summary rows
    for i in [len(table_data) - 2, len(table_data) - 1]:
        for j in range(7):
            table[(i, j)].set_facecolor('#FFE082')
            table[(i, j)].set_text_props(weight='bold')

    plt.title(f'Subject {subject_id} - Optimal Cutoff Frequency Summary',
              fontsize=14, fontweight='bold', pad=20)

    plt.savefig(os.path.join(output_dir, f'subject_{subject_id}_summary_table.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # Create bar plot of optimal cutoffs
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Optimal cutoffs by motion
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(df_summary)))
    bars = ax.bar(range(len(df_summary)), df_summary['Optimal_Cutoff_Hz'], color=colors)
    ax.axhline(y=df_summary['Optimal_Cutoff_Hz'].mean(), color='r',
               linestyle='--', linewidth=2, label=f"Mean: {df_summary['Optimal_Cutoff_Hz'].mean():.1f} Hz")
    ax.set_xlabel('Motion Number', fontweight='bold')
    ax.set_ylabel('Optimal Cutoff (Hz)', fontweight='bold')
    ax.set_title(f'Subject {subject_id}: Optimal Cutoff by Motion', fontweight='bold')
    ax.set_xticks(range(len(df_summary)))
    ax.set_xticklabels([f"{x:02d}" for x in df_summary['Motion_Number']], rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()

    # Plot 2: Composite scores
    ax = axes[1]
    ax.bar(range(len(df_summary)), df_summary['Composite_Score'], color=colors)
    ax.axhline(y=df_summary['Composite_Score'].mean(), color='r',
               linestyle='--', linewidth=2, label=f"Mean: {df_summary['Composite_Score'].mean():.3f}")
    ax.set_xlabel('Motion Number', fontweight='bold')
    ax.set_ylabel('Composite Score', fontweight='bold')
    ax.set_title(f'Subject {subject_id}: Analysis Quality by Motion', fontweight='bold')
    ax.set_xticks(range(len(df_summary)))
    ax.set_xticklabels([f"{x:02d}" for x in df_summary['Motion_Number']], rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'subject_{subject_id}_summary_plots.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Summary plots saved: {output_dir}")


# ============================================================================
# MAIN FUNCTION - MODIFIED FOR BATCH ANALYSIS
# ============================================================================

def main():
    """
    Main function - analyzes all motions for one subject
    """
    # ========== CONFIGURATION ==========
    SUBJECT_ID = 12  # Change this to analyze different subjects
    BVH_DIR = "../../data/bvh_files"
    OUTPUT_DIR = "../../results/cutoff_analysis_multi"
    MOTION_MAPPING_FILE = "../../data/common_motion_mapping.json"
    # ===================================

    print(f"\n{'=' * 80}")
    print(f"BATCH CUTOFF FREQUENCY ANALYSIS")
    print(f"{'=' * 80}")
    print(f"\nConfiguration:")
    print(f"  Subject ID: {SUBJECT_ID}")
    print(f"  BVH Directory: {BVH_DIR}")
    print(f"  Output Directory: {OUTPUT_DIR}")
    print(f"{'=' * 80}\n")

    # Run batch analysis
    df_results = analyze_all_motions(
        subject_id=SUBJECT_ID,
        bvh_dir=BVH_DIR,
        output_dir=OUTPUT_DIR,
        motion_mapping_file=MOTION_MAPPING_FILE
    )

    if df_results is not None:
        print(f"\n{'=' * 80}")
        print("FINAL SUMMARY")
        print(f"{'=' * 80}\n")
        print(df_results.to_string(index=False))
        print(f"\n{'=' * 80}")
        print(
            f"Mean optimal cutoff: {df_results['Optimal_Cutoff_Hz'].mean():.2f} ± {df_results['Optimal_Cutoff_Hz'].std():.2f} Hz")
        print(f"Range: {df_results['Optimal_Cutoff_Hz'].min():.1f} - {df_results['Optimal_Cutoff_Hz'].max():.1f} Hz")
        print(f"{'=' * 80}\n")

        print(f"✅ Analysis complete!")
        print(f"📁 Results saved in: {os.path.join(OUTPUT_DIR, f'subject_{SUBJECT_ID}')}")
        print(f"📈 Summary plots: subject_{SUBJECT_ID}_summary_*.png")
    else:
        print("\n❌ Analysis failed - no results generated")


if __name__ == "__main__":
    main()
