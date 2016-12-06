import numpy as np


class EulerIntegrator(object):
    def __init__(self):
        self.integrands = {}
        self.effectors = []
        self.variables = []
        self.results = {}
        self.ini_values = {}
        self.t_max = None
        self.dt = None
        self.n_steps = None

    def add_integrand(self, integrand):
        affected_variable = integrand.affected_variable
        if affected_variable not in self.integrands:
            # add the new integrand name into the dictionary
            self.integrands[affected_variable] = integrand
            # add the integrand's affected variable into the integrator's
            # variable dictionary
            self._add_variables(integrand.input_variables)
        else:
            expression = "Duplicate key"
            message = "Variable '{0}' is already affected by another integrand in the integrand dictionary"
            raise (IntegrandError(expression, message.format(affected_variable)))

    def add_integrands(self, *integrands):
        for integrand in integrands:
            self.add_integrand(integrand)

    def add_effector(self, effector):
        self.effectors.append(effector)
        self._add_variables(effector.input_variables)

    def _add_variables(self, variables):
        for variable in variables:
            if variable not in self.variables:
                self.variables.append(variable)

    def add_effectors(self, *effectors):
        for effector in effectors:
            self.add_effector(effector)

    def integrate(self, t_max, dt):
        self.t_max = t_max
        self.dt = dt
        self.results["time"], self.n_steps = self._get_time_steps()
        self._preallocate_result_arrays()
        self._initialize_at_zero(self.ini_values)
        for i in range(1,self.n_steps):
            self._integrate_to_step(i)
            self._apply_effectors(i)

    def _apply_effectors(self, step):
        temp_result_dict = {}
        for effector in self.effectors:
            if effector.active:
                if effector.time >= self.results["time"][step] or effector.time == -1:
                    # get effector input variables
                    input_variables = self._get_input_variables(effector, step)
                    affected_variable = effector.affected_variable
                    if affected_variable == "Ie":
                        pass
                    # store result of the effector calculation in a temporary
                    # dictionary with values labeled with variable names
                    temp_result_dict[affected_variable] = effector.function(*input_variables)

                if effector.time != -1:
                    # deactivate used effector
                    effector.active = False

        # replace values in result arrays with ones from the temporary dictionary
        for variable, value in temp_result_dict.items():
            self.results[variable][step] = value

    def _get_input_variables(self, carrier, step):
        input_variables = tuple((self.results[variable][step] if variable[0] != "_"
                                 else self.results[variable][0])
                                # TODO change to -1 maybe if events should be able to change constants
                                for variable in carrier.input_variables)
        return input_variables

    def _get_time_steps(self):
        # get number of time steps for result array pre-allocation
        n_steps = int(np.floor(self.t_max / self.dt) + 1)
        timesteps = np.linspace(0, self.t_max, n_steps)
        return timesteps, n_steps

    def _integrate_to_step(self, step):
        for name, integrand in self.integrands.items():
            # get needed variables as tuple
            input_variables = self._get_input_variables(integrand, step - 1)

            # calculate new value with current integrand function
            new_value = (self.dt * integrand.function(*input_variables) +
                         self.results[integrand.affected_variable][step - 1])
            # update results with calculated value at this time step
            self.results[integrand.affected_variable][step] = new_value

    def _preallocate_result_arrays(self):
        for variable in self.variables:
            # exclude the time array from pre-allocation, the time array already exists at this point
            if variable != "time":
                if variable[0] != "_":
                    self.results[variable] = np.zeros(self.n_steps)
                else:
                    self.results[variable] = np.array([self.ini_values[variable]])

    def _initialize_at_zero(self, step):
        for variable, value in self.ini_values.items():
            if variable in self.results:
                self.results[variable][0] = value
            else:
                self.results[variable] = np.array([value])


class Integrand(object):
    def __init__(self, function, affected_variable, input_variables):
        self.function = function
        self.affected_variable = affected_variable
        self.input_variables = input_variables


class Effector(object):
    def __init__(self, func, affected_variable, input_variables, time):
        self.function = func
        self.affected_variable = affected_variable
        self.input_variables = input_variables
        self.time = time
        self.active = True


class IntegrandError(Exception):
    """Exception raised for errors with integrands

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message
