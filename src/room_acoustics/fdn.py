import random
import numpy as np 
import scipy as sp

from utils import db2lin

from numpy.typing import NDArray
from typing import List
from room_acoustics.analysis import rt2slope


class FeedbackDelayNetwork:
    def __init__(self, sample_rate: int, delay_lengths: List, feedback_matrix_type: str, t60: float = 1.0):
        """
        Initialize the Feedback Delay Network with delay times and feedback matrix.

        Parameters
        ----------
        sample_rate : int
            The sample rate (in Hz).
        delay_lengths : list of int
            List of delay lengths in samples.
        feedback_matrix_type : str
            The type of feedback matrix to use. Options: 'identity', 'orthogonal', 'Hadamard', 'Householder', 'circulant'.
        t60 : float, optional
            Reverberation time in seconds (default is 1.0).
        """
        self.sample_rate = sample_rate
        self.N = len(delay_lengths)
        self.delay_lengths = delay_lengths
        self.t60 = t60
        self.feedback_matrix = self.get_feedback_matrix(feedback_matrix_type)
        self.input_gains = np.random.randn(1, self.N)  # Default input gains for each delay line
        self.output_gains = np.random.randn(self.N, 1) 
       
        # initialize the buffers
        self.delay_buffers = [np.zeros(length) for length in delay_lengths]
        self.write_indices = [0] * self.N
        self.output = np.zeros(0)

    def get_feedback_matrix(self, feedback_matrix_type: str) -> NDArray:
        """
        Generate the feedback matrix based on the specified type.

        Parameters
        ----------
        feedback_matrix_type : str
            The type of feedback matrix to generate.

        Returns
        -------
        NDArray
            The generated feedback matrix.
        """
        # convert to all lower case
        feedback_matrix_type = feedback_matrix_type.lower()

        if feedback_matrix_type == 'identity':
            ## WRITE YOUR CODE HERE ##
            Q = np.eye(self.N) 
        elif feedback_matrix_type == 'random':
            # this is one way to generate a random orthogonal matrix based on QR decomposition
            A = np.random.randn(self.N, self.N)
            Q, R = np.linalg.qr(A)
            Q = np.matmul(Q, np.diag(np.sign(np.diag(R)))) 
        elif feedback_matrix_type == 'hadamard':
            ## WRITE YOUR CODE HERE ##
            raise NotImplementedError("Hadamard feedback matrix generation is not implemented yet.") 
        elif feedback_matrix_type == 'householder':
            ## WRITE YOUR CODE HERE ##
            raise NotImplementedError("Householder feedback matrix generation is not implemented yet.") 
        elif feedback_matrix_type == 'circulant':
            v = np.random.randn(self.N)
            R = np.fft.fft(v)
            R = R / np.abs(R)
            r = np.fft.ifft(R).reshape(-1, 1)  # Ensure r is a column vector
            rnd_sign = 1 if random.random() < 0.5 else -1

            if rnd_sign == 1:
                r2 = np.roll(np.flip(r), 1)
                Q = sp.linalg.toeplitz(r2, r)
            elif rnd_sign == -1:
                r2 = np.roll(r, 1)
                C = sp.linalg.toeplitz(r2, np.flip(r))
                Q = np.fliplr(C)
            else:
                raise ValueError('Not defined')
        else:
            raise ValueError("Invalid feedback matrix type specified.")
        
        self.feedback_matrix = Q

        # apply attenuation 
        gamma = np.power(db2lin(rt2slope(self.t60, self.sample_rate)), np.array(self.delay_lengths))

        Gamma = np.diag(gamma)
        return np.matmul(Gamma, self.feedback_matrix)
    
    def process(self, input_signal: NDArray) -> NDArray:    
        """
        Process the input signal through the Feedback Delay Network.

        Parameters
        ----------
        input_signal : NDArray
            The input audio signal (1D array).

        Returns
        -------
        NDArray
            The processed output signal after passing through the FDN.
        """
        output_signal = []
        
        # process each sample individually
        ### khaled printing
        # print(self.write_indices)
        # print(f"Input signal length: {len(input_signal)}")
        # print(f"delay buffer: {len(self.delay_buffers)}")
        


        ### khaledt printing
        # print(f"Input singnal: {input_signal}")
        for sample_idx, sample in enumerate(input_signal):
            ### WRITE YOUR CODE HERE ###
            # print(f"   -- Current write indices: {self.write_indices}")
            # print(f"current delay buffer values: {[self.delay_buffers[i][self.write_indices[i]] for i in range(self.N)]}")
            # read output from the delay lines
            s_n_current_delays = np.array([self.delay_buffers[i][self.write_indices[i]] for i in range(self.N)])

            # print(f"Sample {sample_idx} :Current delay outputs: {s_n_current_delays}")
            ## END: read output from the delay lines
    
            # compute the new input ´delay_input´ to the delay lines 
            # print(f"   -- Feedback matrix shape: {self.feedback_matrix.shape}")
            # print(f"   -- Current delays shape: {np.array(s_n_current_delays).shape}")
            # print(f"   -- Input gains shape: {self.input_gains.shape}")
            s_n_m_new_delays = np.matmul(self.feedback_matrix, s_n_current_delays) + sample * self.input_gains
            # print(f"   -- New delay input shape: {s_n_m_new_delays.shape}")
            # print(f" new delay input: {s_n_m_new_delays}")
            # print(f" new delay input shape: {s_n_m_new_delays.shape}")
            # print(f"   -- New input to delay lines: {s_n_m_new_delays}")
            ## END: compute the new input ´delay_input´ to the delay lines

           
            for i in range(self.N):
                # store ´delay_input´ in the delay buffers
                self.delay_buffers[i][self.write_indices[i]] = s_n_m_new_delays[0, i]
            # print(f"newly inputted delay buffer values: {[self.delay_buffers[print_idx][self.write_indices[print_idx]] for print_idx in range(self.N)]}")

                ## END: store ´delay_input´ in the delay buffers

            # update the write index for each delay line
            for i in range(self.N):
                self.write_indices[i] = (self.write_indices[i] + 1) % self.delay_lengths[i]
            # print(f"   -- Updated write indices: {self.write_indices}")
            # print(f"   -- post write inx update delay buffers: {[self.delay_buffers[print_idx][self.write_indices[print_idx]] for print_idx in range(self.N)]}")
            ## END: update the write index for each delay line
            
                
            # compute the output sample by multiplying the feedback input with the output gains
            # print(f"   -- Output gains shape: {self.output_gains.shape}")
            # print(f"   -- s_n_current_delays shape: {s_n_current_delays.shape}")
            ## END: compute the output sample
            output_sample = np.matmul(s_n_current_delays, self.output_gains)
            # print(f"   -- Output sample shape: {output_sample.shape}")
            output_signal.append(output_sample[0].item()) # Append the output sample to the output signal
            # print(f"   -- Output signal: {output_signal}")
            # print(f"   -- Output sample shape: {output_sample.shape}")
            # you can the "append" method to store the output samples
            ## END store output sample

        self.output = np.array(output_signal)
        return self.output