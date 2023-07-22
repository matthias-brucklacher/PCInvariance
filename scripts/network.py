"""Deep Hebbian Predictive Coding networks.

"""

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import os
import torch
import torch.nn as nn

class pclayer(nn.Module):
    def __init__(self, layer_size, nextlayer_size, i_rate, device, l_rate):
        """Predictive Coding layer with representation and error neurons. 
            
        Args:
            layer_size (int): Number of neurons in this area.
            nextlayer_size (int): Number of neurons in the higher area.
            i_rate (float): Inference rate.
            device (torch.device): Device that the area is stored on, typically 'cuda:0'.
            l_rate (float): Learning rate.

        """
        super(pclayer, self).__init__() 
        self.device = device 
        self.weights = torch.normal(0, 0.5, (nextlayer_size, layer_size), dtype=torch.float, device=self.device)
        self.weights *=  1/nextlayer_size # Normalize weights
        self.weights = torch.clamp(self.weights, min=0) # No negative weights
        self.actfun = nn.Sigmoid()
        self.inference_rate = i_rate # The inference rate determines the size of activation updates
        self.learn_rate = l_rate # The learning rate determines the size of weight updates

    def forward(self, bu_errors, x_activation, y_activity, nextlayer_y_activity):
        """Perform a single inference step for this area. 

        Args:
            bu_errors (torch.Tensor): Error neuron activity in the area below multiplied by the bottom-up weights.
            x_activation (torch.Tensor): Activation state (membrane potential) of this area.
            y_activity (torch.Tensor): Output activity (firing rate) of representation neurons in this area.
            nextlayer_y_activity (torch.Tensor): Firing rate of representation neurons in higher area.

        Returns:
            e_activity (torch.Tensor): Error neuron activity in this area.
            x_activation (torch.Tensor): New activation state (membrane potential) of this area.
            y_activity (torch.Tensor): New firing rate of representation neurons in this area.

        """
        e_activity = y_activity - torch.matmul(torch.transpose(self.weights, 0, 1), nextlayer_y_activity) # The activity of error neurons is representation - prediction.
        x_activation = x_activation + self.inference_rate * (bu_errors - e_activity) # Inference step: Modify activity depending on error
        y_activity = self.actfun(x_activation - 3) # Apply the (shifted) activation function to get neuronal output
        return e_activity, x_activation, y_activity

    def w_update(self, e_activity, nextlayer_activity):
        """Learning step for weights from this area to the higher area.

        Args:
            e_activity (torch.Tensor): Error neuron activity in this area.
            nextlayer_activity (torch.Tensor): Firing rate of representation neurons in higher area.

        """
        a = self.learn_rate*torch.matmul(nextlayer_activity.reshape(-1,1), e_activity.reshape(1,-1),)
        self.weights = torch.clamp(self.weights + a, min = 0) # Keep only positive weights
    
    def w_batchupdate(self, delta_w):
        """Takes averaged weight updates and applies them to weigths, then clamps to positive values.

        Args:
            delta_w (torch.Tensor): Averaged weight updates.
        """
        self.weights = torch.clamp(self.weights + self.learn_rate*delta_w, min = 0)

class input_layer(pclayer):
    """Input layer. This layer is special as it does not use a full inference step as it is driven only by input (representation neurons are not affected by the top-down signal).

    """
    def forward(self, input, nextlayer_y_activity):
        """Perform a single inference step in the input area. 
        
        Args:
            input (torch.Tensor): Input pattern, a single static image, or a batch thereof.
            nextlayer_y_activity (torch.Tensor): Firing rate of representation neurons in higher area.
        
        Returns:
            e_activity (torch.Tensor): Error neuron activity in this area.

        """
        e_activity = input - torch.matmul(torch.transpose(self.weights, 0, 1), nextlayer_y_activity)
        return e_activity

class output_layer(pclayer):
    """Topmost layer. This layer requires a different inference step as no top-down predictions exist.

    """
    def forward(self, bu_errors, x_activation):
        x_activation = x_activation + self.inference_rate*bu_errors
        y_activity = self.actfun(x_activation-3)
        return x_activation, y_activity

class Network(nn.Module):
    def __init__(self, size_list, i_rate1, i_rate2, i_rate3, device, l_rate):
        """Predictive coding network consisting of multiple predictive coding areas.

        Args:
            size_list (list): List of integer values containing the number of neurons per area.
            i_rate1 (float): Inference rate area 1.
            i_rate2 (float): Inference rate area 2.
            i_rate3 (float): Inference rate area 3.
            device (torch.device): Device on which the network is stored, typically 'cuda:0'.
            l_rate (float): Learning rate.

        """
        self.device = device
        super(Network, self).__init__()
        self.size_list = size_list
        self.input_layer = input_layer(size_list[0], size_list[1], 0, self.device, l_rate)
        self.pclayer1 = pclayer(size_list[1], size_list[2], i_rate1, self.device, l_rate)
        self.pclayer2 = pclayer(size_list[2], size_list[3], i_rate2, self.device, l_rate)
        self.output_layer = output_layer(size_list[3], 1 , i_rate3, self.device, l_rate) # Output layer connects to a layer with a single neuron that does not influence it. Could also rewrite constructor...
    
    def load_weights(self, path_to_weights):
        """Load pretrained weights.

        Args:
            path_to_weights (str): Directory containing the pretrained weights.

        """
        #assert [os.path.exists(path_to_weights + weightfile) for weightfile in ['w1.pt', 'w2.pt', 'w3.pt']] == [True, True, True], f'Weights to be loaded not in correct path. Expected at {path_to_weights}'
        self.input_layer.weights = torch.load(path_to_weights + 'w1.pt').to(self.device)
        self.pclayer1.weights = torch.load(path_to_weights + 'w2.pt').to(self.device)
        self.pclayer2.weights = torch.load(path_to_weights + 'w3.pt').to(self.device)
        print("Successfully loaded pretrained weights from path " + path_to_weights)

    def save_weights(self, directory):
        """Save network weights.

        Args:
            directory (str): Directory in which to store the weights.

        """
        if not os.path.exists(directory):
            os.mkdir(directory)
        else:    
            print("Directory " , directory ,  " already exists")
        torch.save(self.input_layer.weights, directory+'/w1.pt')
        torch.save(self.pclayer1.weights, directory+'/w2.pt')
        torch.save(self.pclayer2.weights, directory+'/w3.pt')

    def init_act(self, size_list):
        """Initialize neuron states (membrane potentials/activations).

        Args:
            size_list (list): List containing the number of neurons per area.

        Returns:
            e0 (torch.Tensor): All zero error neuron activity in the input area.
            e1 (torch.Tensor): All zero error neuron activity in area 1.
            x1 (torch.Tensor): Uniform membrane potential representation neurons of area 1
            y1 (torch.Tensor): Uniform firing rate representation neurons of area 1.
            e2 (torch.Tensor): All zero error neuron activity in area 2.
            x2 (torch.Tensor): Uniform membrane potential representation neurons of area 2.
            y2 (torch.Tensor): Uniform firing rate representation neurons of area 2.
            x3 (torch.Tensor): Uniform membrane potential representation neurons of area 3.
            y3 (torch.Tensor): Uniform firing rate representation neurons of area 3.

        """
        e0 = torch.zeros(size_list[0]).to(self.device)
        e1 = torch.zeros(size_list[1]).to(self.device)
        x1 = -2*torch.ones(size_list[1]).to(self.device)
        y1 = self.pclayer1.actfun(x1-3).to(self.device)
        e2 = torch.zeros(size_list[2]).to(self.device)
        x2 = -2*torch.ones(size_list[2]).to(self.device)
        y2 = self.pclayer1.actfun(x2-3).to(self.device)
        x3 = -2*torch.ones(size_list[3]).to(self.device)
        y3 = self.output_layer.actfun(x3-3).to(self.device)
        return e0, e1, x1, y1, e2, x2, y2, x3, y3

    def init_batches(self, batch_size):
        """Initialize neuron states (membrane potentials/activations) in batches.

        Args:
            size_list (list): List containing the number of neurons per area.

        Returns:
            e0 (torch.Tensor): All zero error neuron activity in the input area.
            e1 (torch.Tensor): All zero error neuron activity in area 1.
            x1 (torch.Tensor): Uniform membrane potential representation neurons of area 1
            y1 (torch.Tensor): Uniform firing rate representation neurons of area 1.
            e2 (torch.Tensor): All zero error neuron activity in area 2.
            x2 (torch.Tensor): Uniform membrane potential representation neurons of area 2.
            y2 (torch.Tensor): Uniform firing rate representation neurons of area 2.
            x3 (torch.Tensor): Uniform membrane potential representation neurons of area 3.
            y3 (torch.Tensor): Uniform firing rate representation neurons of area 3.

        """
        e0_batch = torch.zeros(self.size_list[0], batch_size).to(self.device)
        e1_batch = torch.zeros(self.size_list[1], batch_size).to(self.device)
        x1_batch = -2*torch.ones(self.size_list[1], batch_size).to(self.device)
        y1_batch = self.pclayer1.actfun(x1_batch - 3).to(self.device)
        e2_batch = torch.zeros(self.size_list[2], batch_size).to(self.device)
        x2_batch = -2*torch.ones(self.size_list[2], batch_size).to(self.device)
        y2_batch = self.pclayer1.actfun(x2_batch -3 ).to(self.device)
        x3_batch = -2*torch.ones(self.size_list[3], batch_size).to(self.device)
        y3_batch = self.output_layer.actfun(x3_batch - 3).to(self.device)
        return e0_batch, e1_batch, x1_batch, y1_batch, e2_batch, x2_batch, y2_batch, x3_batch, y3_batch

    def forward(self, frame, inference_steps, e0, e1, x1, y1, e2, x2, y2, x3, y3):
        """Perform a number of inference steps on a given input for the whole network.

        Args:
            frame (torch.Tensor): Static input frame.
            inference_steps (int): Number of inference steps to perform.
            e0 (torch.Tensor): Error neuron activity in the input area.
            e1 (torch.Tensor): Error neuron activity in area 1.
            x1 (torch.Tensor): Membrane potential representation neurons of area 1
            y1 (torch.Tensor): Firing rate representation neurons of area 1.
            e2 (torch.Tensor): Error neuron activity in area 2.
            x2 (torch.Tensor): Membrane potential representation neurons of area 2.
            y2 (torch.Tensor): Firing rate representation neurons of area 2.
            x3 (torch.Tensor): Membrane potential representation neurons of area 3.
            y3 (torch.Tensor): Firing rate representation neurons of area 3.

        Returns:
            e0 (torch.Tensor): Error neuron activity in the input area.
            e1 (torch.Tensor): Error neuron activity in area 1.
            x1 (torch.Tensor): Membrane potential representation neurons of area 1
            y1 (torch.Tensor): Firing rate representation neurons of area 1.
            e2 (torch.Tensor): Error neuron activity in area 2.
            x2 (torch.Tensor): Membrane potential representation neurons of area 2.
            y2 (torch.Tensor): Firing rate representation neurons of area 2.
            x3 (torch.Tensor): Membrane potential representation neurons of area 3.
            y3 (torch.Tensor): Firing rate representation neurons of area 3.

        """
        for i in range(inference_steps):
            e0 = self.input_layer(frame, y1)
            e1, x1, y1 = self.pclayer1(torch.matmul(self.input_layer.weights, e0), x1, y1, y2)
            e2, x2, y2 = self.pclayer2(torch.matmul(self.pclayer1.weights, e1), x2, y2, y3)
            x3, y3 = self.output_layer(torch.matmul(self.pclayer2.weights, e2), x3)
        return e0, e1, x1, y1, e2, x2, y2, x3, y3

    def learn(self, e0, e1, e2, y1, y2, y3):
        """Update weigths in each layer based on pre- and postsynaptic activity.
        
        Args:
            e0 (torch.Tensor): Error neuron activity in the input area.
            e1 (torch.Tensor): Error neuron activity in area 1.
            y1 (torch.Tensor): Firing rate representation neurons of area 1.
            e2 (torch.Tensor): Error neuron activity in area 2.
            y2 (torch.Tensor): Firing rate representation neurons of area 2.
            y3 (torch.Tensor): Firing rate representation neurons of area 3.

        """
        self.input_layer.w_update(e0, y1)
        self.pclayer1.w_update(e1, y2)
        self.pclayer2.w_update(e2, y3)

    def reconstruction_error(self, test_sequences, inference_steps):
        """Compute recostruction error averaged over data. Also returns the inferred layer-3 representations as numpy array for further use.
        
        Args:
            test_sequences: Torch array of sequences for which representations shall be inferred. 
                Shape is (n_classes * n_instances_per_class * n_trafos, n_frames_per_sequence, n_neurons)
            inference_steps: Number of inference steps to be executed on each frame
        
        Returns:
            reconstruction_error1(numpy.ndarray):
            reconstruction_error2(numpy.ndarray):
            reconstruction_error3(numpy.ndarray):
            y1_reps (numpy.ndarray): Inferred representations area 1. Shape is 
            y2_reps (numpy.ndarray): Inferred representations area 2. Shape is
            y3_reps (numpy.ndarray): Inferred representations area 3. Shape is
            
        """
        # Reshape data to 2d for batch inference
        n_classes, n_instances_per_class, n_trafos, n_frames_per_seq, n_neurons = test_sequences.shape[0], test_sequences.shape[1], test_sequences.shape[2], test_sequences.shape[3], test_sequences.shape[4]
        n_total_frames = n_classes * n_instances_per_class * n_trafos * n_frames_per_seq
        data = test_sequences.flatten(start_dim=0, end_dim=3) # -> Shape (n_total_frames, n_neurons) 
        data = torch.transpose(data, 0, 1) # -> Shape (n_neurons, n_frames)
        batch_size = n_total_frames
        e0_batch, e1_batch, x1_batch, y1_batch, e2_batch, x2_batch, y2_batch, x3_batch, y3_batch = self.init_batches(batch_size) # Reset network
        e0_batch, e1_batch, x1_batch, y1_batch, e2_batch, x2_batch, y2_batch, x3_batch, y3_batch = self(data, inference_steps, e0_batch, e1_batch, x1_batch, y1_batch, e2_batch, x2_batch, y2_batch, x3_batch, y3_batch)

        # Reconstructions from area 1        
        reconstruction_error1 = torch.mean(torch.square(e0_batch)).cpu().numpy()
        
        # Reconstructions from area 2 all the way down
        recon_2 = torch.matmul(torch.transpose(self.pclayer1.weights, 0, 1), y2_batch)
        recon_2 = torch.matmul(torch.transpose(self.input_layer.weights, 0, 1), recon_2)
        reconstruction_error2 = torch.mean(torch.square(recon_2 - data)).cpu().numpy()

        # Reconstructions from area 2 all the way down
        recon_3 = torch.matmul(torch.transpose(self.pclayer2.weights, 0, 1), y3_batch)
        recon_3 = torch.matmul(torch.transpose(self.pclayer1.weights, 0, 1), recon_3)
        recon_3 = torch.matmul(torch.transpose(self.input_layer.weights, 0, 1), recon_3)
        reconstruction_error3 = torch.mean(torch.square(recon_3 - data)).cpu().numpy()

        # Revert dimensions of y3_batch from [number of neurons, batch_size] to [batch_size, number of neurons] and then to [seq_num, frames_per_seq, number of neurons]
        y1_reps = torch.transpose(y1_batch, 0, 1).reshape((n_classes * n_instances_per_class * n_trafos, n_frames_per_seq, self.size_list[-3])).cpu().numpy()
        y2_reps = torch.transpose(y2_batch, 0, 1).reshape((n_classes * n_instances_per_class * n_trafos, n_frames_per_seq, self.size_list[-2])).cpu().numpy()
        y3_reps = torch.transpose(y3_batch, 0, 1).reshape((n_classes * n_instances_per_class * n_trafos, n_frames_per_seq, self.size_list[-1])).cpu().numpy()
        
        return reconstruction_error1, reconstruction_error2, reconstruction_error3, y1_reps, y2_reps, y3_reps

    def reconstruct_topdown(self, frame, i_steps, save_as):
        """Takes an input image, runs a number of inference steps, then reconstructs the input top-down from all levels in the network hierarchy.

        Args:
            frame (torch.Tensor): Static input pattern from which to infer, and which to reconstruct.
            i_steps (int): Number of inference steps. Should be at least 2000.
            save_as (str): Path under which to store the plotted image and reconstructions.

        """
        # Initialize activity and conduct inference on frame
        e0, e1, x1, y1, e2, x2, y2, x3, y3 = self.init_act(self.size_list) 
        e0, e1, x1, y1, e2, x2, y2, x3, y3 = self(frame, i_steps, e0, e1, x1, y1, e2, x2, y2, x3, y3) # 

        # Calculate reconstructions by multiplying representations with weight matrices and applzing activation function 
        rep_1 = y1
        recon_1 = torch.matmul(torch.transpose(self.input_layer.weights, 0, 1), rep_1)
        
        rep_2 = y2
        recon_2 = torch.matmul(torch.transpose(self.pclayer1.weights, 0, 1), rep_2)
        recon_2 = torch.matmul(torch.transpose(self.input_layer.weights, 0, 1), recon_2)

        rep_3 = y3  
        recon_3 = torch.matmul(torch.transpose(self.pclayer2.weights, 0, 1), rep_3)
        recon_3 = torch.matmul(torch.transpose(self.pclayer1.weights, 0, 1), recon_3)
        recon_3 = torch.matmul(torch.transpose(self.input_layer.weights, 0, 1), recon_3)
        
        # Plot original input and reconstructions from layer 1, layer 2, layer 3.
        maxvalue = torch.max(torch.stack([frame, recon_1, recon_2, recon_3], dim = 0))
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows = 1, ncols = 4)
        data_width = int(np.sqrt(frame.shape[-1]))
        im = ax1.matshow(frame.reshape(data_width, data_width).cpu(), vmin = 0, vmax = maxvalue)
        ax1.set_title('Original')
        im2 = ax2.matshow(recon_1.reshape(data_width, data_width).cpu(), vmin = 0, vmax = maxvalue)
        ax2.set_title("Area 1")
        im3 = ax3.matshow(recon_2.reshape(data_width, data_width).cpu(), vmin = 0, vmax = maxvalue)
        ax3.set_title("Area 2")
        im4 = ax4.matshow(recon_3.reshape(data_width, data_width).cpu(), vmin = 0, vmax = maxvalue)
        ax4.set_title("Area 3")
        for ax in [ax1,ax2,ax3,ax4]:
            ax.set_xticks([])
            ax.set_yticks([])
        axins = inset_axes(ax4,
                    width="5%",  
                    height="100%",
                    loc='right',
                    borderpad=-2
                   )
        fig.colorbar(im, cax = axins)#, orientation = "horizontal") #cax=cbar_ax)
        plt.savefig(save_as)
        plt.close()

    def convergence_over_isteps(self, data, timesteps):
        """Measure convergence of representation neurons in all areas across inference time steps.

        Args:
            timesteps (int): Number of inference steps.
            data (torch.Tensor): Input data. Shape is (n_classes * n_instances * n_trafos, n_frames_per_sequence, img_height*img_width).

        Returns:
            y_updates (dict): Dictionary with a list of update values per RN population (length is given by argument timesteps).

        """
        assert data.dim() == 3, 'Data has wrong shape.'
        batch_size = data.shape[0]*data.shape[1]
        data = torch.transpose(data.reshape(data.shape[0]*data.shape[1], data.size()[-1]),0,1) # Batch dimension must precede
        y_updates = {'y1': [], 'y2': [], 'y3': []}
        e0_batch, e1_batch, x1_batch, y1_batch, e2_batch, x2_batch, y2_batch, x3_batch, y3_batch = self.init_batches(batch_size) # Reset network
        
        # Initialize network
        for timestep in range(timesteps):
            y1_prev, y2_prev, y3_prev = torch.clone(y1_batch), torch.clone(y2_batch), torch.clone(y3_batch)# Clone previous activities
            e0_batch, e1_batch, x1_batch, y1_batch, e2_batch, x2_batch, y2_batch, x3_batch, y3_batch = self(data, 1, e0_batch, e1_batch, x1_batch, y1_batch, e2_batch, x2_batch, y2_batch, x3_batch, y3_batch) # Conduct inference step
        
            # Compute differences and append.
            y_updates['y1'].append(torch.mean(y1_prev- y1_batch).cpu().numpy())
            y_updates['y2'].append(torch.mean(y2_prev- y2_batch).cpu().numpy())
            y_updates['y3'].append(torch.mean(y3_prev- y3_batch).cpu().numpy())
            
        return y_updates