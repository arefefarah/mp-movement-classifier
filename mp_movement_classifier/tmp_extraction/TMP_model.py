#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 13:34:12 2018
Code for Temporal movement primitives
@author: dom
"""

# base package
import torch.nn
import torch.distributions
import torch.optim

import scipy.signal
import torch_hessian
import numpy
# from movement_primitives.movement_classifier.src.torch_hessian import *
from collections import OrderedDict

# from plotting import *

import unittest


class MP_model(torch.nn.Module):
    """Synchronous temporal movement primitive model that can learn and predict from varying segment lengths and can handle pretty big data"""


    def __init__(self,num_t_points,num_MPs,num_signals=None,num_segments=None,kernel_width=10.0,kernel_var=0.4**2,noise_level=0.03,init_data=None, gpu = False):
        """Initialize MP model.
        num_t_points: number of time discretization points onto which the GP for the MPs is conditioned, i.e. the effective parametrization
        num_MPs: number of MPs
        num_signals: number of sensors/signals. May be None, if init_data is supplied
        num_segments: number of segments. May be None, if init_data is supplied
        kernel_width: initial RBF kernel width for MPs, in time discretization points. estimated from autocorrelation function of init_data
        kernel_var: initial RBF kernel variance. should match signal variance
        noise_level: noise variance
        init_data=list(segment_data[signals,time]) data from which to initalize the MPs.
        Either init_data, init_segment_starts or num_signals, kernel_width, kernel_var have to be supplied
        """

        super(MP_model,self).__init__()

        if gpu:
            torch.set_default_tensor_type('torch.cuda.DoubleTensor')
            self.device = torch.device('cuda:0')
        else:
            torch.set_default_tensor_type('torch.DoubleTensor')
            self.device = torch.device('cpu')
            
        self.num_t_points=num_t_points
        self.num_MPs=num_MPs
        self.gpu = gpu

        zero=torch.tensor(0.0)
        zeros=torch.zeros(num_t_points)

        # Laplace prior on the weights, for sparser solutions
        #self.weight_prior=torch.distributions.laplace.Laplace(zero,1.0)

        # i.i.d. Gaussian prior on the weights
        self.weight_prior=torch.distributions.normal.Normal(zero,1.0)

        # Gaussian process prior with RBF Kernel on the MPs.
        if init_data is not None:
            # self.init_model_params(init_data)

            # PCA on subset of data
            self.init_model_params_subset(init_data, subset_size=300)
            print("Initial kernel variance", self.kernel_var)
            print("Initial kernel width",self.kernel_width)
        else:
            self.kernel_var=kernel_var
            self.kernel_width=kernel_width

        if gpu:
            x=torch.arange(self.num_t_points)
        else:
            x=numpy.arange(self.num_t_points)
            
        self.K= self.kernel_matrix(x,x,self.kernel_var,self.kernel_width)
        
        if gpu:
            self.invK = torch.inverse(self.K)
            self.MP_prior=torch.distributions.multivariate_normal.MultivariateNormal(zeros,covariance_matrix=self.K.clone().detach())
        else:    
            self.invK = numpy.linalg.inv(self.K)
            self.MP_prior=torch.distributions.multivariate_normal.MultivariateNormal(zeros,covariance_matrix=torch.tensor(self.K,dtype=torch.float64))

        
        if init_data is None:
            self.init_model_params_random(num_segments,num_signals)

        # observation model, a simple Gaussian with stddev 0.03
        self.noise_model=torch.distributions.normal.Normal(zero,noise_level)
        self.noise_level = noise_level

        # dictionary of resampling segment length -> projection matrix for that resampling
        if gpu:
            self.resampling_matrix=OrderedDict()
        else:
            self.resampling_matrix=dict()


        self.curP=None
        self.cur_VAF=None


    def kernel_matrix(self,x,y,variance,width):
        """Compute kernel matrix between time point arrays x,y having supplied mean and variance"""
        if x.shape!=y.shape or (x!=y).any():
            if self.gpu:
                return variance * torch.exp( -0.5*(x.reshape(-1, 1) - y)**2/(width**2) )
            else:               
                return variance * numpy.exp( -0.5*(numpy.subtract.outer(x,y))**2/(width**2) )
        else:
            if self.gpu:
                return variance * torch.exp( -0.5*(x.reshape(-1, 1) - y)**2/(width**2) ) + 1e-6*torch.eye(len(x))
            else:                   
                return variance * numpy.exp( -0.5*(numpy.subtract.outer(x,y))**2/(width**2) ) + 1e-6*numpy.eye(len(x))


    def _get_params_as_tensors(self):
        all_mps=torch.stack(list(self.MPs))
        all_weights=torch.stack(list(self.weights))
        return all_mps,all_weights


    def _get_mps_as_tensor(self):
        return torch.stack(list(self.MPs))


    def predict(self,segment_lengths,all_mps=None,as_numpy=False):
        """Predict data by multiplying weights with MPs and resampling the predictions to fill segments with supplied ends"""
        if all_mps is None:
            all_mps=self._get_mps_as_tensor()

        resampled_predictions=[]

        for segidx in range(len(self.weights)):

            seg_len=segment_lengths[segidx]
            resampled_predictions.append(self.predict_one_segment(seg_len,segidx,all_mps,as_numpy))

        return resampled_predictions


    def predict_one_segment(self,seg_len,segidx,all_mps=None,as_numpy=False):
        """Predict one segment of length seg_len with supplied weights[signal,MP]"""
        if all_mps is None:
            all_mps=self.get_mps_as_tensor()

        if seg_len not in self.resampling_matrix:
            if self.gpu:
                t = torch.mm(self.kernel_matrix(torch.arange(seg_len)*(self.num_t_points/seg_len),
                                                                                          torch.arange(self.num_t_points),self.kernel_var,self.kernel_width), self.invK)
                self.resampling_matrix[seg_len]=t.clone().detach()
    
            else:
                self.resampling_matrix[seg_len]=torch.tensor(numpy.dot(self.kernel_matrix(numpy.arange(seg_len)*(self.num_t_points/seg_len),numpy.arange(self.num_t_points),self.kernel_var,self.kernel_width),self.invK))

        pred_seg=torch.tensordot(torch.mm(self.weights[segidx],all_mps),self.resampling_matrix[seg_len],dims=((1,),(1,)))
        if as_numpy:
            return pred_seg.detach().numpy()
        return pred_seg


    def log_prob_segment(self,segment,segidx,all_mps=None,getPredError=False):
        """Compute log-probability of one segment[sensor,time] segment and weights under their prior, using the weights from segment segidx.
        Also returns prediction error, if getPredError is True"""
        if all_mps is None:
            all_mps=self.get_mps_as_tensor()

        pred_error=segment-self.predict_one_segment(segment.shape[1],segidx,all_mps) # J x T
        
        
        ll=self.noise_model.log_prob(pred_error).sum()+self.weight_prior.log_prob(self.weights[segidx]).sum()

        if getPredError:
            if self.gpu:
                return ll,pred_error.detach()
            else:
                return ll,pred_error.detach().numpy()
        else:
            return ll


    def joint_prob_data_model(self,data):
        """Joint probability of data[sensor,time] and model parameters given the end points of each segment"""
        all_mps=self._get_mps_as_tensor()

        self.cur_log_p_joint=0.0
        pred_error=[]

        for segidx in range(len(data)):
            dseg=data[segidx]
            ll,pe=self.log_prob_segment(dseg,segidx,all_mps,getPredError=True)
            self.cur_log_p_joint+=ll
            pred_error.append(pe)

        self.cur_log_p_joint+=self.MP_prior.log_prob(all_mps).sum()
        
        if self.gpu:
            pred_error=torch.cat(pred_error, 1)**2 # J x T
            variances=[(t.var(dim=1).mean()) for t in data]
            variance = torch.mean(torch.stack(variances))
            self.cur_VAF=float(1.0-pred_error.mean()/variance)
                
        else:
            pred_error=numpy.hstack(pred_error)**2 # J x T
            variances=[float(t.var(dim=1).mean()) for t in data] # for segment (t) in data
            variance=numpy.mean(variances)
            self.cur_VAF=float(1.0-pred_error.mean()/variance)


        return self.cur_log_p_joint
            

    def init_model_params(self,init_data):
        """Init model params by principal component analysis (PCA)
        data[sensor,time], segment_end[idx] end of segments."""

        sensors=None
        segments=[]
        kernel_width_estimates=[]
        for segment in init_data:

            if sensors is None:
                sensors=len(segment)
            else:
                if sensors!=len(segment):
                    raise ValueError("All segments must have the same number of sensors, but found {0:d} and {1:d}".format(sensors,len(segment)))

            # resample segment to self.num_t_points
            slen2=segment.shape[1] // 2
            ctr=numpy.hstack([numpy.ones((1,slen2))*segment[:,0:1],segment,numpy.ones((1,slen2))*segment[:,-2:-1]]) # pad start and end to avoid resampling artifacts
            resampled_segment=scipy.signal.resample(ctr,self.num_t_points*2,axis=1)[:,self.num_t_points//2:3*self.num_t_points//2] # resample and cut out the middle
            segments.append(resampled_segment)

            # estimate RBF kernel width
            sd=numpy.hstack(resampled_segment)
            cf=(numpy.correlate(sd,sd,mode="full")/len(sd)-sd.mean()**2).clip(0.0)
            cf=cf[len(cf)//2-self.num_t_points//2:len(cf)//2+self.num_t_points//2] # cut off noisy bits of distribution that would mess up the variance estimate
            cf/=cf.sum() # normalize autocorrelation to make it interpretable as a probability density
            kernel_width_estimates.append(numpy.sqrt((cf*numpy.arange(-len(cf)//2,len(cf)//2)**2).sum())) # compute variance, which is kernel width. Factor 2 for finite-sample-size correction


        self.kernel_width=numpy.mean(kernel_width_estimates)

        concat_segments=numpy.concatenate(segments,axis=0)
        self.kernel_var=concat_segments.var(axis=1).mean()

        U,S,V=numpy.linalg.svd(concat_segments)
        self.weights=torch.nn.ParameterList()
        # we're keeping weights and MPs in separate lists so that they can be used separately as leaf variables
        # for automatic 2nd derivatives to keep the hessian from overflowing the memory
        for trial in range(len(segments)):
            self.weights.append(torch.nn.Parameter(torch.tensor(U[trial*sensors:(trial+1)*sensors,:self.num_MPs]),requires_grad=True))

        self.MPs=torch.nn.ParameterList()
        all_mps=numpy.dot(numpy.diag(S[:self.num_MPs]),V[:self.num_MPs])
        for mp in range(self.num_MPs):
            self.MPs.append(torch.nn.Parameter(torch.tensor(all_mps[mp]),requires_grad=True))

    def init_model_params_subset(self, init_data, subset_size=50, random_seed=42):
        """Init model params by PCA on a subset of the data to avoid memory issues

        Args:
            init_data: list of segments [sensor, time]
            subset_size: number of segments to use for PCA initialization
            random_seed: for reproducible subset selection
        """

        # Set random seed for reproducible results
        numpy.random.seed(random_seed)

        # Select a random subset of segments
        num_segments = len(init_data)
        if subset_size >= num_segments:
            # Use all data if subset_size is larger than available data
            selected_indices = list(range(num_segments))
            print(f"Using all {num_segments} segments for initialization")
        else:
            selected_indices = numpy.random.choice(num_segments, size=subset_size, replace=False)
            print(f"Using {subset_size} out of {num_segments} segments for PCA initialization")

        subset_data = [init_data[i] for i in selected_indices]

        # Now use the existing logic but only on the subset
        sensors = None
        segments = []
        kernel_width_estimates = []

        for segment in subset_data:
            if sensors is None:
                sensors = len(segment)
            else:
                if sensors != len(segment):
                    raise ValueError(
                        f"All segments must have the same number of sensors, but found {sensors} and {len(segment)}")

            # Resample segment to self.num_t_points
            slen2 = segment.shape[1] // 2
            ctr = numpy.hstack([
                numpy.ones((1, slen2)) * segment[:, 0:1],
                segment,
                numpy.ones((1, slen2)) * segment[:, -2:-1]
            ])
            resampled_segment = scipy.signal.resample(ctr, self.num_t_points * 2, axis=1)[:,
                                self.num_t_points // 2:3 * self.num_t_points // 2]
            segments.append(resampled_segment)

            # Estimate RBF kernel width
            sd = numpy.hstack(resampled_segment)
            cf = (numpy.correlate(sd, sd, mode="full") / len(sd) - sd.mean() ** 2).clip(0.0)
            cf = cf[len(cf) // 2 - self.num_t_points // 2:len(cf) // 2 + self.num_t_points // 2]
            cf /= cf.sum()
            kernel_width_estimates.append(numpy.sqrt((cf * numpy.arange(-len(cf) // 2, len(cf) // 2) ** 2).sum()))

        self.kernel_width = numpy.mean(kernel_width_estimates)

        # PCA on the subset only
        concat_segments = numpy.concatenate(segments, axis=0)
        self.kernel_var = concat_segments.var(axis=1).mean()

        U, S, V = numpy.linalg.svd(concat_segments)

        # Initialize weights for ALL segments (not just the subset)
        self.weights = torch.nn.ParameterList()

        # For segments in the subset, use PCA weights
        subset_weights = {}
        for i, seg_idx in enumerate(selected_indices):
            subset_weights[seg_idx] = U[i * sensors:(i + 1) * sensors, :self.num_MPs]

        # For all segments (including those not in subset), initialize weights
        for seg_idx in range(len(init_data)):
            if seg_idx in subset_weights:
                # Use PCA-derived weights for subset segments
                weight_init = torch.tensor(subset_weights[seg_idx])
            else:
                # Use average of PCA weights plus small random noise for non-subset segments
                avg_weights = numpy.mean([subset_weights[idx] for idx in subset_weights.keys()], axis=0)
                noise = numpy.random.normal(0, 0.1, avg_weights.shape)
                weight_init = torch.tensor(avg_weights + noise)

            self.weights.append(torch.nn.Parameter(weight_init, requires_grad=True))

        # Initialize MPs from PCA
        self.MPs = torch.nn.ParameterList()
        all_mps = numpy.dot(numpy.diag(S[:self.num_MPs]), V[:self.num_MPs])
        for mp in range(self.num_MPs):
            self.MPs.append(torch.nn.Parameter(torch.tensor(all_mps[mp]), requires_grad=True))

        print(f"Model initialized with kernel_var={self.kernel_var:.4f}, kernel_width={self.kernel_width:.4f}")
    def init_model_params_random(self,num_segments,num_signals):
        """Init model params randomly for num_trials many trials and num_signals many signals per trial"""

        # we're keeping weights and MPs in separate lists so that they can be used separately as leaf variables
        # for automatic 2nd derivatives to keep the hessian from overflowing the memory
        self.weights=torch.nn.ParameterList()
        for trial in range(num_segments):
            self.weights.append(torch.nn.Parameter(self.weight_prior.sample((num_signals,self.num_MPs)),requires_grad=True))

        self.MPs=torch.nn.ParameterList()
        for mp in range(self.num_MPs):
            self.MPs.append(torch.nn.Parameter(self.MP_prior.sample(),requires_grad=True))


    def sample(self):
        """Sample data from an initialized model. Does not (yet) enforce boundary constraints."""
        new_weights=[]
        segment_lengths=[]
        for w in self.weights:
            new_weights.append(self.weight_prior.sample(w.shape))
            segment_lengths.append(numpy.random.randint(self.num_t_points//2,2*self.num_t_points))

        self.weights,new_weigths=new_weights,self.weights
        segments=self.predict(segment_lengths)
        self.weights,new_weigths=new_weights,self.weights

        return segments


    def require_grad_all(self,rg=True):
        for df in list(self.MPs)+list(self.weights):
            df.requires_grad_(rg)


    def Laplace_approx_segment(self,segment,segidx):
        """Compute model evidence contribution of one segment"""
        seg_weights=self.weights[segidx]

        self.require_grad_all(False)
        self.weights[segidx].requires_grad(True)
        logp_joint=self.log_prob_segment(self,data,segidx)
        invCV= torch_hessian.hessian(logp_joint, [self.weights[segidx]])
        sign,logdet=numpy.linalg.slogdet(prec_block)

        logp=logp_joint+seg_weights.numel()*numpy.log(2.0*numpy.pi)/2.0-logdet/2.0
        self.require_grad_all(True)

        return float(logp)


    def Laplace_approx(self,data):
        """Compute laplace approximation to log(P(D|model)) assuming that current weights/MPs have been optimized,
        and gradients have been cleared"""
        data=[torch.tensor(d, device = self.device) for d in data]

        logp_joint=self.joint_prob_data_model(data)

        # big-data approximation to the precision matrix: for many datapoints (trials), the
        # MPs will be almost certain. That implies that the weights are nearly independent across trials,
        # because the path from the weights of one trial to another is a tail-to-tail path through the MPs
        # thus, we can compute the determinant of the hessian block-wise, which should avoid memory overflow
        logdeth=0.0
        tot_num_df=0
        all_dfs=list(self.MPs)+list(self.weights)

        self.require_grad_all(False)
        for df in all_dfs:
            tot_num_df+=df.numel()


        i=0
        for df in all_dfs:
            print("Laplacian: at df",i,"of",len(all_dfs),"with number of elements",df.numel())
            i+=1
            df.requires_grad_(True)
            logp_joint=self.joint_prob_data_model(data)
            prec_block= torch_hessian.hessian(logp_joint, [df])
            sign,logdet=numpy.linalg.slogdet(prec_block)
            logdeth+=logdet
            df.requires_grad_(False)

        self.require_grad_all(True)


        # Laplace approximation
        logp=logp_joint+tot_num_df*numpy.log(2.0*numpy.pi)/2.0-logdeth/2.0
        
        self.LAP = float(logp)
        
        return float(logp)


    def forward(self,data):
        """Overwritten from torch.nn.Module.forward so we can use the usual calling syntax"""
        neg_log_p_joint=-self.joint_prob_data_model(data) # Forward pass: Compute predicted y by passing x to the model
        return neg_log_p_joint


    def learn_one_segment(self,segment,segidx,max_steps=100):
        """Learn weights for segment with index segidx"""

        all_mps=self._get_mps_as_tensor()

        self.require_grad_all(False)
        self.weights[segidx].requires_grad_(True)

        optimizer=torch.optim.LBFGS(self.weights[segidx]) # for a gaussian prior and obs model, this will be exact after signals*MP many steps

        def closure():
            optimizer.zero_grad()
            neg_log_p=-self.log_prob_segment(segment,segidx,all_mps)
            neg_log_p.backward()
            return neg_log_p

        for step in range(self.weights[segidx].numel()):
            optimizer.step(closure)

            gradlen=torch.norm(self.weights[segidx].flatten())
            if gradlen<1e-4:
                break

        self.require_grad_all(True)


    def learn(self,data,adam_steps=1000,bfgs_steps=1000):
        # data=[torch.tensor(d) for d in data] #there is a valuerror here which is resolved by making a copy 
        data = [torch.tensor(d.copy()) for d in data]

        optimizer1=torch.optim.Adam(self.parameters())
        optimizer2=torch.optim.LBFGS(self.parameters())

        self.learn_curve=[self.joint_prob_data_model(data).detach()]
        self.VAF_curve=[self.cur_VAF]

        def closure():
            optimizer.zero_grad()
            neg_log_p=self(data) # get loss  # This calls forward()
            neg_log_p.backward() # compute gradient of the loss with respect to model parameters
            return neg_log_p

        if bfgs_steps is None:
            all_dfs=list(self.MPs)+list(self.weights)
            bfgs_steps=0
            for df in all_dfs:
                bfgs_steps+=df.numel()


        # the learning iteration
        gradlen0=None
        for step in range(adam_steps+bfgs_steps):

            if step<adam_steps:
                optimizer=optimizer1
            else:
                optimizer=optimizer2

            optimizer.step(closure) # update parameters

            self.learn_curve.append(self.cur_log_p_joint.detach())
            self.VAF_curve.append(self.cur_VAF)


            # stop learning when gradient is small enough
            grad=torch.cat([w.grad.flatten() for w in self.weights]+[s.grad.flatten() for s in self.MPs])
            gradlen=torch.norm(grad)
            if gradlen0 is None:
                gradlen0=gradlen

            if (step<adam_steps and step % 100 == 0) or (step>adam_steps):
                print("Learning step",step,"at P=",float(self.learn_curve[-1]),", VAF=",self.VAF_curve[-1],", rel. gradient length=",float(gradlen/gradlen0))
            if gradlen/gradlen0<5e-4:
                break

        optimizer.zero_grad()



class TestTMPModel(unittest.TestCase):

    def setUp(self):
        """Generate ground truth data from a random teacher model"""

        # self.num_timepoints_per_primitive=50
        # self.num_ground_truth_segments=500
        # self.num_ground_truth_signals=11
        # self.num_ground_truth_MPs=5
        
        # self.teacher=MP_model(self.num_timepoints_per_primitive,self.num_ground_truth_MPs,self.num_ground_truth_signals,
        #                       self.num_ground_truth_segments,kernel_width=10.0)
        # self.segment_lengths=numpy.random.randint(self.num_timepoints_per_primitive//2,2*self.num_timepoints_per_primitive,size=self.num_ground_truth_segments)
        # self.ground_truth_data=self.teacher.predict(self.segment_lengths,as_numpy=True)
        # plot_kernel(self.teacher.K)
        # plot_mp(torch.stack(list(self.teacher.MPs)),"teacher, rand. init")
        ########################################################

        """Use real BVH data instead of generating ground truth data"""
        destination_folder = "../../data/MMpose/segmented_files/hand_waving/bvh_files"
        folder_path = "../../../data/MMpose/bvh_files" 
        folder_path = destination_folder  # if specific motion is assumed for training only
        # folder_path = "../../BVH_small"
        bvh_data ,_ = read_bvh_files(folder_path)
        
        # Process data according to paper specifications
        self.ground_truth_data = process_bvh_data(bvh_data)
        
        # Set model parameters based on your data
        self.num_timepoints_per_primitive = 30
        self.num_ground_truth_MPs = 10
        
        # Find how many signals are in your data (assuming all segments have same number of signals)
        self.num_ground_truth_signals = self.ground_truth_data[0].shape[0]
        
        # Store the number of segments
        self.num_ground_truth_segments = len(self.ground_truth_data)
        
        # Get segment lengths from your data
        self.segment_lengths = numpy.array([segment.shape[1] for segment in self.ground_truth_data])
        
        # You can still plot the initial data if you want
        # If you want to visualize one of the segments:
        if len(self.ground_truth_data) > 0:
            sample_seg = self.ground_truth_data[0]
            plt.figure(figsize=(10, 6))
            for i in range(min(5, sample_seg.shape[0])):  # Plot up to 5 signals
                plt.plot(sample_seg[i], label=f'Signal {i}')
            plt.title('Sample segment from BVH data')
            plt.legend()
            plt.show()

        # self.teacher=MP_model(self.num_timepoints_per_primitive,self.num_ground_truth_MPs,self.num_ground_truth_signals,
        #                       self.num_ground_truth_segments,kernel_width=10.0)
        self.teacher=MP_model(self.num_timepoints_per_primitive,self.num_ground_truth_MPs,
                              init_data=self.ground_truth_data)
      
        
        plot_kernel(self.teacher.K)
        plot_mp(torch.stack(list(self.teacher.MPs)),"teacher, pca_real dat init")

    # def test1LearningPCA(self):
    #     """Test learning of a model that has as many MPs as the teacher and is PCA-initalized"""

    #     for student_timepoints in [self.num_timepoints_per_primitive,self.num_timepoints_per_primitive//2,self.num_timepoints_per_primitive*2]:

    #         student=MP_model(student_timepoints,self.num_ground_truth_MPs,init_data=self.ground_truth_data)

    #         student.learn(self.ground_truth_data,1000,10)

    #         lc=student.learn_curve
    #         vc=student.VAF_curve
    #         epochs=numpy.arange(len(lc))

    #         plot_learn_curve(epochs,lc,vc,"student_{0:d}, PCA init".format(student_timepoints),save=True)
    #         plot_learn_curve(epochs[-50:],lc[-50:],vc[-50:],"student_{0:d}, PCA init, tail".format(student_timepoints),save=True)

    #         recon_data=student.predict(self.segment_lengths,as_numpy=True)

    #         plot_reconstructions(self.ground_truth_data[self.segment_lengths.argmax()],recon_data[self.segment_lengths.argmax()],f"student_{student_timepoints}, max,PCA,",save=True)
    #         plot_reconstructions(self.ground_truth_data[self.segment_lengths.argmin()],recon_data[self.segment_lengths.argmin()],f"student_{student_timepoints}, min,PCA,",save=True)
    #         plot_mp(torch.stack(list(student.MPs)),"student_{0:d}, PCA init".format(student_timepoints),save=True)

    #         self.assertTrue(vc[-1]>0.98,"VAF after learning is only {0:f}, check convergence!".format(vc[-1]))


    def test2LearningRandom(self):
        """Test learning of a model that has as many MPs as the teacher and is randomly initalized"""

        for student_timepoints in [self.num_timepoints_per_primitive,self.num_timepoints_per_primitive//2,self.num_timepoints_per_primitive*2]:

            student=MP_model(student_timepoints,self.num_ground_truth_MPs,self.num_ground_truth_signals,self.num_ground_truth_segments,kernel_width=10.0)

            student.learn(self.ground_truth_data,2000,20)

            lc=student.learn_curve
            vc=student.VAF_curve
            epochs=numpy.arange(len(lc))

            plot_learn_curve(epochs,lc,vc,"student_{0:d}, rand init".format(student_timepoints),save=True)
            plot_learn_curve(epochs[-50:],lc[-50:],vc[-50:],"student_{0:d}, rand init, tail".format(student_timepoints),save=True)

            recon_data=student.predict(self.segment_lengths,as_numpy=True)

            plot_reconstructions(self.ground_truth_data[self.segment_lengths.argmax()],recon_data[self.segment_lengths.argmax()],"student_{0:d}, rand,max".format(student_timepoints),save=True)
            plot_reconstructions(self.ground_truth_data[self.segment_lengths.argmin()],recon_data[self.segment_lengths.argmin()],"student_{0:d}, rand,min".format(student_timepoints),save=True)
            plot_mp(torch.stack(list(student.MPs)),"student_{0:d}, rand init".format(student_timepoints),save=True)

            self.assertTrue(vc[-1]>0.98,"VAF after learning is only {0:f}, check convergence!".format(vc[-1]))


    def test3ModelComparison(self):
        """Model selection test"""

        student_timepoints=self.num_timepoints_per_primitive//2

        model_evidences=[]
        VAFs=[]
        for student_num_MPs in range(1,10):

            student=MP_model(student_timepoints,student_num_MPs,self.num_ground_truth_signals,self.num_ground_truth_segments,kernel_width=10.0)

            student.learn(self.ground_truth_data,adam_steps=2000)

            VAFs.append(student.VAF_curve[-1])
            # Compute laplace approximation to log(P(D|model)) --> posterior
            model_evidences.append(student.Laplace_approx(self.ground_truth_data))


        plot_model_comparison(model_evidences,VAFs,ground_truth_num_MPs=self.num_ground_truth_MPs,title="")

        model_evidences=numpy.array(model_evidences)
        best_num_MPs=model_evidences.argmax()+1

        self.assertTrue(numpy.abs(best_num_MPs-self.num_ground_truth_MPs)<2,"Model comparison found num.MP={0:d}, but it should be at num.MP={1:d}".format(best_num_MPs,self.num_ground_truth_MPs))





if __name__=="__main__":


    unittest.main()
