import samna
from samna.speck2e.configuration import SpeckConfiguration
from sinabs.backend.dynapcnn.dynapcnn_layer import DynapcnnLayer
from sinabs.backend.dynapcnn.config_builder import ConfigBuilder # Import ConfigBuilder
import torch                                                    # For tensor operations
import sinabs                                                   # For sinabs.activation types
from sinabs.backend.dynapcnn.dvs_layer import expand_to_pair    # For expand_to_pair utility

# Removed: from .dynapcnn import DynapcnnConfigBuilder


class Speck2EConfigBuilder(ConfigBuilder): # Inherit from ConfigBuilder
    @classmethod
    def get_samna_module(cls):
        return samna.speck2e

    @classmethod
    def get_default_config(cls) -> "SpeckConfiguration":
        return SpeckConfiguration()

    @classmethod
    def get_input_buffer(cls):
        return samna.BasicSourceNode_speck2e_event_speck2e_input_event()

    @classmethod
    def get_output_buffer(cls):
        return samna.BasicSinkNode_speck2e_event_output_event()

    @classmethod
    def set_kill_bits(cls, layer: DynapcnnLayer, config_dict: dict) -> dict:
        """
        Speck2e specific implementation for setting kill bits.
        In this version, it simply returns the config_dict without modification,
        implying Speck2e handles kill bits differently or they are not configured
        through this mechanism in the same way DYNAP-CNN does.
        """
        return config_dict

    @classmethod
    def get_dynapcnn_layer_config_dict(cls, layer: DynapcnnLayer):
        """
        Generates the configuration dictionary for a DynapcnnLayer,
        adapted for Speck2e.
        This method's content is based on the DynapcnnConfigBuilder's
        get_dynapcnn_layer_config_dict, ensuring necessary configurations
        are generated. The call to cls.set_kill_bits() at the end will
        invoke the Speck2EConfigBuilder.set_kill_bits() method defined above.
        """
        config_dict = {}
        config_dict["destinations"] = [{}, {}]  # Default empty destinations

        # Update the dimensions
        channel_count, input_size_y, input_size_x = layer.input_shape
        dimensions = {"input_shape": {}, "output_shape": {}}
        dimensions["input_shape"]["size"] = {"x": input_size_x, "y": input_size_y}
        dimensions["input_shape"]["feature_count"] = channel_count

        (f, h, w) = layer.get_neuron_shape()
        dimensions["output_shape"]["size"] = {}
        dimensions["output_shape"]["feature_count"] = f
        dimensions["output_shape"]["size"]["x"] = w
        dimensions["output_shape"]["size"]["y"] = h
        dimensions["padding"] = {
            "x": layer.conv_layer.padding[1],
            "y": layer.conv_layer.padding[0],
        }
        dimensions["stride"] = {
            "x": layer.conv_layer.stride[1],
            "y": layer.conv_layer.stride[0],
        }
        dimensions["kernel_size"] = layer.conv_layer.kernel_size[0]

        if dimensions["kernel_size"] != layer.conv_layer.kernel_size[1]:
            raise ValueError("Conv2d: Kernel must have same height and width.")
        config_dict["dimensions"] = dimensions

        # Update parameters from convolution
        if layer.conv_layer.bias is not None:
            (weights, biases) = layer.conv_layer.parameters()
        else:
            (weights,) = layer.conv_layer.parameters()
            biases = torch.zeros(layer.conv_layer.out_channels)
        
        # Transpose weights to match samna convention (H, W, C_in, C_out) -> (C_out, C_in, H, W) then (C_out, C_in, W, H)
        # PyTorch weights are (out_channels, in_channels, kernel_height, kernel_width)
        # Samna expects (out_features, in_features, kernel_width, kernel_height)
        # Assuming layer.conv_layer.parameters() gives PyTorch order.
        weights = weights.transpose(2, 3) # (C_out, C_in, H, W) -> (C_out, C_in, W, H) for samna
        config_dict["weights"] = weights.int().tolist()
        config_dict["biases"] = biases.int().tolist()
        config_dict["leak_enable"] = biases.bool().any()

        # Update parameters from the spiking layer
        # - Neuron states
        if not layer.spk_layer.is_state_initialised():
            f_neuron, h_neuron, w_neuron = layer.get_neuron_shape() # Use neuron shape for state
            neurons_state = torch.zeros(f_neuron, w_neuron, h_neuron) # Samna neuron state order: (features, width, height)
        elif layer.spk_layer.v_mem.dim() == 4: # (batch, features, height, width)
            # PyTorch v_mem: (batch, features, height, width)
            # Samna neurons_initial_value: (features, width, height)
            neurons_state = layer.spk_layer.v_mem.transpose(2, 3)[0] # Get (features, width, height)
        else:
            raise ValueError(
                f"Current v_mem (shape: {layer.spk_layer.v_mem.shape}) of spiking layer not understood."
            )

        # - Resetting vs returning to 0
        if isinstance(layer.spk_layer.reset_fn, sinabs.activation.MembraneReset):
            return_to_zero = True
        elif isinstance(layer.spk_layer.reset_fn, sinabs.activation.MembraneSubtract):
            return_to_zero = False
        else:
            raise Exception(
                "Unknown reset mechanism. Only MembraneReset and MembraneSubtract are currently understood."
            )

        if layer.spk_layer.min_v_mem is None:
            min_v_mem = -(2**15)  # Default DYNAP-CNN min potential if not specified
        else:
            min_v_mem = int(layer.spk_layer.min_v_mem)
            
        config_dict.update(
            {
                "return_to_zero": return_to_zero,
                "threshold_high": int(layer.spk_layer.spike_threshold),
                "threshold_low": min_v_mem,
                "monitor_enable": False, # Default, can be changed by other methods
                "neurons_initial_value": neurons_state.int().tolist(),
            }
        )

        # Update parameters from pooling
        if layer.pool_layer is not None:
            # Assuming destination[0] is the primary one for pooling configuration
            # expand_to_pair might return (val, val) if given an int
            pooling_val = expand_to_pair(layer.pool_layer.kernel_size)[0]
            if pooling_val > 1 : # Only set pooling if it's actual pooling
                 config_dict["destinations"][0]["pooling"] = pooling_val
                 config_dict["destinations"][0]["enable"] = True # Typically pooling implies destination is active
            # If pooling is 1, it's like no pooling, destination might be set by other logic (e.g., routing)
        # else: pass # No pooling layer, destinations remain default or set by routing logic

        # Set kill bits by calling this class's set_kill_bits method.
        # For Speck2EConfigBuilder, this method (as defined above) will simply return config_dict.
        config_dict = cls.set_kill_bits(layer=layer, config_dict=config_dict)

        return config_dict