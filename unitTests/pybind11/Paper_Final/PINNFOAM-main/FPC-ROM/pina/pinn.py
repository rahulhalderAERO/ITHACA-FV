""" Module for PINN """
import torch

from .problem import AbstractProblem
from .label_tensor import LabelTensor
import numpy
from scipy.io import savemat
import time


torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732


class PINN(object):

    def __init__(self,
            problem,
            model,
            optimizer=torch.optim.Adam,
            lr=0.001,
            regularizer=0.00001,
            dtype=torch.float32,
            device='cpu',
            error_norm='mse'):
        '''
        :param Problem problem: the formualation of the problem.
        :param torch.nn.Module model: the neural network model to use.
        :param float lr: the learning rate; default is 0.001.
        :param float regularizer: the coefficient for L2 regularizer term.
        :param type dtype: the data type to use for the model. Valid option are
            `torch.float32` and `torch.float64` (`torch.float16` only on GPU);
            default is `torch.float64`.
        '''

        if dtype == torch.float64:
            raise NotImplementedError('only float for now')

        self.problem = problem
        # self.getM = problem.getM()
        # self.getK = problem.getK()        
        self.rand_choice_integer_Eq = problem.rand_choice_integer_Eq()
        self.rand_choice_integer_Data = problem.rand_choice_integer_Data()

        # self._architecture = architecture if architecture else dict()
        # self._architecture['input_dimension'] = self.problem.domain_bound.shape[0]
        # self._architecture['output_dimension'] = len(self.problem.variables)
        # if hasattr(self.problem, 'params_domain'):
            # self._architecture['input_dimension'] += self.problem.params_domain.shape[0]

        self.error_norm = error_norm

        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError
        self.device = torch.device(device)

        self.dtype = dtype
        self.history_loss = {}

        self.model = model
        self.model.to(dtype=self.dtype, device=self.device)

        self.truth_values = {}
        self.input_pts = {}

        self.trained_epoch = 0
        self.optimizer = optimizer(
            self.model.parameters(), lr=lr, weight_decay=regularizer)

    @property
    def problem(self):
        return self._problem

    @problem.setter
    def problem(self, problem):
        if not isinstance(problem, AbstractProblem):
            raise TypeError
        self._problem = problem

    def _compute_norm(self, vec):
        """
        Compute the norm of the `vec` one-dimensional tensor based on the
        `self.error_norm` attribute.

        .. todo: complete

        :param vec torch.tensor: the tensor
        """
        if isinstance(self.error_norm, int):
            return torch.linalg.vector_norm(vec, ord = self.error_norm,  dtype=self.dytpe)
        elif self.error_norm == 'mse':
            return torch.mean(vec.pow(2))
        elif self.error_norm == 'me':
            return torch.mean(torch.abs(vec))
        else:
            raise RuntimeError

    def save_state(self, filename):

        checkpoint = {
                'epoch': self.trained_epoch,
                'model_state': self.model.state_dict(),
                'optimizer_state' : self.optimizer.state_dict(),
                'optimizer_class' : self.optimizer.__class__,
                'history' : self.history_loss,
                'input_points_dict' : self.input_pts,
        }        
        torch.save(checkpoint, filename)

    def load_state(self, filename):

        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model_state'])


        self.optimizer = checkpoint['optimizer_class'](self.model.parameters())
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])

        self.trained_epoch = checkpoint['epoch']
        self.history_loss = checkpoint['history']

        self.input_pts = checkpoint['input_points_dict']

        return self
    
    def span_tensor_given_pts(self, *args, **kwargs):
        arguments = args[0]
        print(arguments)
        locations = kwargs.get('locations', 'all')        
        if locations == 'all':
            locations = [condition for condition in self.problem.conditions]
        for location in locations:
            condition = self.problem.conditions[location]
            pts = condition.location.sample_tensor_given_pts(
                    arguments['n'],
                    variables=arguments['variables'])
            self.input_pts[location] = pts  #.double()  # TODO
            self.input_pts[location] = (
                self.input_pts[location].to(dtype=self.dtype,
                                            device=self.device))
            self.input_pts[location].requires_grad_(True)
            self.input_pts[location].retain_grad()


    def span_pts(self, *args, **kwargs):
        """
        >>> pinn.span_pts(n=10, mode='grid')
        >>> pinn.span_pts(n=10, mode='grid', location=['bound1'])
        >>> pinn.span_pts(n=10, mode='grid', variables=['x'])
        """

        def merge_tensors(tensors):  # name to be changed
            if len(tensors) == 2:
                tensor1 = tensors[0]
                tensor2 = tensors[1]
                n1 = tensor1.shape[0]
                n2 = tensor2.shape[0]

                tensor1 = LabelTensor(
                    tensor1.repeat(n2, 1),
                    labels=tensor1.labels)
                tensor2 = LabelTensor(
                    tensor2.repeat_interleave(n1, dim=0),
                    labels=tensor2.labels)
                return tensor1.append(tensor2)
            elif len(tensors) == 1:
                return tensors[0]
            else:
                recursive_result = merge_tensors(tensors[1:])
                return merge_tensors([tensors[0], recursive_result])

        if isinstance(args[0], int) and isinstance(args[1], str):
            argument = {}
            argument['n'] = int(args[0])
            argument['mode'] = args[1]
            argument['variables'] = self.problem.input_variables
            arguments = [argument]
        elif all(isinstance(arg, dict) for arg in args):
            arguments = args
        elif all(key in kwargs for key in ['n', 'mode']):
            argument = {}
            argument['n'] = kwargs['n']
            argument['mode'] = kwargs['mode']
            argument['variables'] = self.problem.input_variables
            arguments = [argument]
        else:
            raise RuntimeError

        locations = kwargs.get('locations', 'all')
        
        
        if locations == 'all':
            locations = [condition for condition in self.problem.conditions]
        for location in locations:
                        
            condition = self.problem.conditions[location]            
            
            pts = merge_tensors([
                condition.location.sample(
                    argument['n'],
                    argument['mode'],
                    variables=argument['variables'])
                for argument in arguments])
                        
            self.input_pts[location] = pts  #.double()  # TODO
            self.input_pts[location] = (
                self.input_pts[location].to(dtype=self.dtype,
                                            device=self.device))
            self.input_pts[location].requires_grad_(True)
            self.input_pts[location].retain_grad()
            
            
    def loss_computation_function(self):
            losses_function = []
            for condition_name in self.problem.conditions:
                condition = self.problem.conditions[condition_name]
                if hasattr(condition, 'function'):
                    pts = self.input_pts[condition_name]
                    predicted = self.model(pts)
                    pts_new = pts[self.model.seq_length+1:,:].as_subclass(LabelTensor)
                    pts_new.labels = self.model.input_variables                    
                    # print ("size of the predicted is ===", predicted.size(), "size of the pts_new is ===", pts_new.size())
                    for function in condition.function:
                        residuals = function(self,pts_new, predicted)
                        # residuals = function(pts, predicted)
                        local_loss = (
                            condition.data_weight*self._compute_norm(
                                residuals))
                        losses_function.append(local_loss)
            return losses_function
    
    
    def loss_computation_output(self):
            losses_output = []
            for condition_name in self.problem.conditions:
                condition = self.problem.conditions[condition_name]
                if hasattr(condition, 'output_points'):
                    
                    pts = condition.input_points                    
                    pts = (pts.to(dtype=self.dtype,device=self.device))
                    pts.requires_grad_(True)
                    pts.retain_grad()
                    predicted = self.model(pts)
                    
                    # MODIFY OUTPUT Points:
                    
                    output_tensor = condition.output_points
                    tensors_y = torch.stack([output_tensor[i+self.model.seq_length] for i in range(len(output_tensor)-self.model.seq_length-1)])                    
                    residuals = (predicted - tensors_y ).reshape(-1,1)
                    local_loss = (
                        condition.data_weight*self._compute_norm(residuals))
                    losses_output.append(local_loss)    
            return losses_output    
                     
    def train(self, stop=100, frequency_print=2, save_loss=1, trial=None):

        epoch = 0

        header = []
        for condition_name in self.problem.conditions:
            condition = self.problem.conditions[condition_name]

            if hasattr(condition, 'function'):
                if isinstance(condition.function, list):
                    for function in condition.function:
                        header.append(f'{condition_name}{function.__name__}')

                    continue

            header.append(f'{condition_name}')

        start_time = time.time()
        print("start_time is ====", start_time)
        while True:            
            losses = []
            
            
            for condition_name in self.problem.conditions:
                condition = self.problem.conditions[condition_name]
                 
                
                if (condition_name == 'A'): 
                 
                 if (epoch == 0) or (epoch > 0 and epoch % 50 == 0): 
                  
                  if hasattr(condition, 'function'):
                
                    pts = self.input_pts[condition_name]
                    predicted = self.model(pts)
                    pts_new = pts
                    pts_new.labels = self.model.input_variables                                        
                    for function in condition.function:
                        Derivative = function(self,pts_new, predicted)
                        Derivative_m1 = Derivative['Residual_derivative_m1'].detach()
                        Derivativep = Derivative['Residualp_derivative_m1'].detach()
                          
                if (condition_name == 'D'):
                
                  if hasattr(condition, 'function'):
                
                    pts = self.input_pts[condition_name]
                    predicted = self.model(pts)
                    pts_new = pts
                    pts_new.labels = self.model.input_variables                    
                    for function in condition.function:                                                
                        y_mid_dict = function(self,pts_new, predicted) 
                        y_mid = y_mid_dict['new_tensor']
                        y_midp = y_mid_dict['new_tensor_p']
                        y_mid_sq = y_mid.pow(2)
                        # L = torch.linalg.torch.mean(y_mid_sq) 
                        # losses.append(L)                       
                        y_mid_d = y_mid.detach()
                        y_midp_d = y_midp.detach()
                        
                        k = 0
                        for i in range(2,predicted.shape[0],100):
                        # for i in range(2,1001,100):                             
                            if i == 2: 
                              dmulty_Mat = torch.mm(Derivative_m1[k*31:(k+1)*31,:], predicted[i,:].reshape(-1,1))
                              dmulty_Mat_p = torch.mm(Derivativep[k*10:(k+1)*10,:], predicted[i,:].reshape(-1,1))                              
                            else: 
                              dmulty_new = torch.mm(Derivative_m1[k*31:(k+1)*31,:], predicted[i,:].reshape(-1,1))
                              dmulty_new_p = torch.mm(Derivativep[k*10:(k+1)*10,:], predicted[i,:].reshape(-1,1))                                
                              dmulty_Mat = torch.cat((dmulty_Mat,dmulty_new), dim = 0)
                              dmulty_Mat_p = torch.cat((dmulty_Mat_p,dmulty_new_p), dim = 0)
                            k = k+1
                        
                        
                        # print("The size of the y_mid_d is ===", y_mid_d.shape)
                        # print("dmulty_Mat is ===", dmulty_Mat.shape)
                        Res_Dis = 0.1*(2*y_mid_d*y_mid - 2*y_mid_d*dmulty_Mat)                        
                        # # ===== Loss Computation Phase I================================================================                                                
                        L_Dis = torch.linalg.torch.mean(Res_Dis)
                        losses.append(L_Dis) 

                                                
                        Res_Dis_p = 0.1*(2*y_midp_d*dmulty_Mat_p)                        
                        # # ===== Loss Computation Phase I================================================================                                                
                        L_Dis_p = torch.linalg.torch.mean(Res_Dis_p)
                        losses.append(L_Dis_p) 
                        
                if hasattr(condition, 'output_points'):
                    
                    pts = condition.input_points                    
                    pts = (pts.to(dtype=self.dtype,device=self.device))
                    pts.requires_grad_(True)
                    pts.retain_grad()
                    predicted = self.model(pts)
                    
                    # print("pts is ===", pts)
                    
                    # MODIFY OUTPUT Points:
                    
                    output_tensor = condition.output_points                    
                    list_arrayD = [0,250,500,750,1000]                   
                    residuals1 = (predicted[list_arrayD,:] - output_tensor[list_arrayD,:])
                    # residuals1 = (predicted[:1001,:] - output_tensor[:1001,:])
                    residuals_aligned = residuals1.reshape(-1,1)
                    local_loss = (
                        condition.data_weight*self._compute_norm(residuals_aligned))
                    losses.append(local_loss)
                
                
                if hasattr(condition, 'output_points'):
                    list_arrayD = [0]                    
                    fixed_number = 1001
                    list_arrayD1 = [x + fixed_number for x in list_arrayD]
                    fixed_number = 1001
                    list_arrayD2 = [x + fixed_number for x in list_arrayD]
                    residuals2 = (predicted[list_arrayD2,:] - output_tensor[list_arrayD1,:])
                    residuals_aligned = residuals2.reshape(-1,1)
                    local_loss = (
                        condition.data_weight*self._compute_norm(residuals_aligned))
                    losses.append(local_loss)
                
                
                
                
                if hasattr(condition, 'output_points'):
                    list_arrayD = [0,250,500,750,1000]                    
                    fixed_number = 2002
                    list_arrayD1 = [x + fixed_number for x in list_arrayD]
                    fixed_number = 2002
                    list_arrayD2 = [x + fixed_number for x in list_arrayD]
                    residuals3 = (predicted[list_arrayD2,:] - output_tensor[list_arrayD1,:])
                    residuals_aligned = residuals3.reshape(-1,1)
                    local_loss = 2*(
                        condition.data_weight*self._compute_norm(residuals_aligned))
                    losses.append(local_loss)
                    
            self.optimizer.zero_grad() 
            # (losses[1]+losses[2]).backward()  
            sum(losses).backward()            
            self.optimizer.step()
            if save_loss and (epoch % save_loss == 0 or epoch == 0):
                self.history_loss[epoch] = [
                    loss.detach().item() for loss in losses]

            if trial:
                import optuna
                trial.report(sum(losses), epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            if isinstance(stop, int):
                if epoch == stop:
                    print('[epoch {:05d}] {:.6e} '.format(self.trained_epoch, sum(losses).item()), end='')
                    for loss in losses:
                        print('{:.6e} '.format(loss.item()), end='')
                    print()
                    break
            elif isinstance(stop, float):
                if sum(losses) < stop:
                    break

            if epoch % frequency_print == 0 or epoch == 1:
                print('       {:5s}  {:12s} '.format('', 'sum'),  end='')
                for name in header:
                    print('{:12.12s} '.format(name), end='')
                print()

                print('[epoch {:05d}] {:.6e} '.format(self.trained_epoch, sum(losses).item()), end='')
                for loss in losses:
                    print('{:.6e} '.format(loss.item()), end='')
                print()

            self.trained_epoch += 1
            epoch += 1
        end_time = time.time()
        total_time = end_time - start_time
        print("end_time is ====", end_time)
        print("total_time is ====", total_time)          
        return sum(losses).item()


    def error(self, dtype='l2', res=100):

        import numpy as np
        if hasattr(self.problem, 'truth_solution') and self.problem.truth_solution is not None:
            pts_container = []
            for mn, mx in self.problem.domain_bound:
                pts_container.append(np.linspace(mn, mx, res))
            grids_container = np.meshgrid(*pts_container)
            Z_true = self.problem.truth_solution(*grids_container)

        elif hasattr(self.problem, 'data_solution') and self.problem.data_solution is not None:
            grids_container = self.problem.data_solution['grid']
            Z_true = self.problem.data_solution['grid_solution']
        try:
            unrolled_pts = torch.tensor([t.flatten() for t in grids_container]).T.to(dtype=self.dtype, device=self.device)
            Z_pred = self.model(unrolled_pts)
            Z_pred = Z_pred.detach().numpy().reshape(grids_container[0].shape)

            if dtype == 'l2':
                return np.linalg.norm(Z_pred - Z_true)/np.linalg.norm(Z_true)
            else:
                # TODO H1
                pass
        except:
            print("")
            print("Something went wrong...")
            print("Not able to compute the error. Please pass a data solution or a true solution")
