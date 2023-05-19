import logging
from typing import Optional

from torch.distributions import Distribution

from goldmine.inference.scandal import SCANDALInference
from goldmine.ml.trainer import train_model
from goldmine.ml.losses import negative_log_likelihood, score_mse
from goldmine.simulators.base import Simulator


class SeqSCANDALInference(SCANDALInference):

    def __init__(self, simulator:Simulator, prior: Optional[Distribution] = None, **params):
        super().__init__(params)

        self.simulator = simulator
        self.prior = prior

    def sequential_train(self, n_epochs, numb_sim, batch_size, lr, device, **kwargs):
        logging.info('Training sequential SCANDAL with the following settings:')
        logging.info('  Epochs:     %s', n_epochs)
        logging.info('  Batch size: %s', batch_size)
        logging.info('  Learning rate: %s', lr)

        self.numb_sim = numb_sim

        # initial samples
        theta = self.prior.sample((batch_size,))

        # TODO Until no better convergence
        for _ in range(n_epochs):
            
            
            # Simulate data
            x = self.simulator.simulate(theta)
        
            # Train
            train_model(
                self.model,
                negative_log_likelihood,
                theta,
                x,
                n_epochs=n_epochs,
                batch_size=batch_size,
                lr=lr,
                device=device,
                **kwargs
            )

            # Get Score function from trained model

            # Generate new samples using SVGD

            # Update theta


            


        
