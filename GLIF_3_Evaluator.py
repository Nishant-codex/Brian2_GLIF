#!/bin/env python

import bluepyopt as bpop
from bluepyopt.parameters import Parameter

import numpy as np

import GLIF_3 as glif_model
import brian2 as b2
b2.prefs.codegen.target = 'cython'

class Evaluator(bpop.evaluators.Evaluator):
    def __init__(self, input_current, dt, init_values, parameters, fitness, target_voltage):
        self.input_current = input_current
        self.dt = dt
        self.init_values = init_values
        self.parameters = parameters
        self.fitness = fitness
        self.target_voltage = target_voltage

        super(Evaluator, self).__init__(
                objectives=[x for x in fitness.keys()],
                params=[
                    Parameter('El',
                        value = parameters['El']['value'],
                        bounds = parameters['El']['bounds'],
                        frozen = parameters['El']['frozen']),
                    Parameter('C',
                        value = parameters['C']['value'],
                        bounds = parameters['C']['bounds'],
                        frozen = parameters['C']['frozen']),
                    Parameter('G',
                        value = parameters['G']['value'],
                        bounds = parameters['G']['bounds'],
                        frozen = parameters['G']['frozen']),
                    Parameter('Th_inf',
                        value = parameters['Th_inf']['value'],
                        bounds = parameters['Th_inf']['bounds'],
                        frozen = parameters['Th_inf']['frozen']),
                    Parameter('t_ref',
                        value = parameters['t_ref']['value'],
                        bounds = parameters['t_ref']['bounds'],
                        frozen = parameters['t_ref']['frozen']),
                    Parameter('V_reset',
                        value = parameters['V_reset']['value'],
                        bounds = parameters['V_reset']['bounds'],
                        frozen = parameters['V_reset']['frozen']),
                    Parameter('A_0',
                        value = parameters['A_0']['value'],
                        bounds = parameters['A_0']['bounds'],
                        frozen = parameters['A_0']['frozen']),
                    Parameter('k_0',
                        value = parameters['k_0']['value'],
                        bounds = parameters['k_0']['bounds'],
                        frozen = parameters['k_0']['frozen']),
                    Parameter('R_0',
                        value = parameters['R_0']['value'],
                        bounds = parameters['R_0']['bounds'],
                        frozen = parameters['R_0']['frozen']),
                    Parameter('A_1',
                        value = parameters['A_1']['value'],
                        bounds = parameters['A_1']['bounds'],
                        frozen = parameters['A_1']['frozen']),
                    Parameter('k_1',
                        value = parameters['k_1']['value'],
                        bounds = parameters['k_1']['bounds'],
                        frozen = parameters['k_1']['frozen']),
                    Parameter('R_1',
                        value = parameters['R_1']['value'],
                        bounds = parameters['R_1']['bounds'],
                        frozen = parameters['R_1']['frozen']),
                    ]
                )

    def evaluate_with_dicts(self, param_dict):
        b2.set_device('cpp_standalone')

        # Simulate with parameter set
        param_dict_units = glif_model.add_parameter_units(param_dict)
        t, V, I_0, I_1,  = glif_model.run_brian_sim(
                self.input_current * b2.amp,
                self.dt * b2.second,
                self.init_values,
                param_dict_units)

        #Evaluate fitness
        fitness = {x: self.fitness[x](t, self.target_voltage, V) for x in self.fitness.keys()}

        b2.device.delete(force = True)
        b2.device.reinit()

        return fitness

    def evaluate_with_lists(self, param_list):
        param_dict = glif_model.parameters_from_list(param_list)
        fitness = self.evaluate_with_dicts(param_dict)
        return np.array([fitness[x] for x in fitness.keys()])
