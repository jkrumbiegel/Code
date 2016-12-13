import numpy as np


class EulerIntegrator(object):
    """Does Euler integration using Integrands, Calculators and Effectors.

    Takes functions contained in Integrands and integrates variables over time.
    In each time step, after integration, additional variables can be computed
    using Calculators, and can be changed according to specified conditions
    using Effectors. The order of operations is always integration,
    calculation,
    and Effector manipulation. Only in the first time step (t=0), there is no
    integration.

    Attributes:
        calculators: A list containing all added Calculator objects.
        integrands: A dictionary containing all added Integrands by their names.
        effectors: A list containing all added Effector objects.
        input_variables: A list of the names of all input variables used by
        Integrands, Calculators and Effectors.
        affected_variables: A list of the names of all variables being changed
            by either Integrands, Calculators, or Effectors.
        results: A dictionary of numpy arrays containing the computed values for
            all variables in affected_variables for all time steps.
        ini_values: A dictionary with numerical values for all input variables
            at time step t=0.
        t_max: A number specifying the end point of the integration in time.
        dt: A number specifying the time step advanced in each integration loop.
        n_steps: A number specifying the number of time steps of the integration
            and thereby also the length of all result arrays, except for
            constants.
    """

    def __init__(self):
        self.calculators = []
        self.integrands = {}
        self.effectors = []
        self.input_variables = []
        self.affected_variables = []
        self.results = {}
        self.ini_values = {}
        self.t_max = None
        self.dt = None
        self.n_steps = None

    def add_integrand(self, integrand):
        """Adds a new Integrand into the integrand list.

        Adds a new Integrand to the Integrator's list of integrands, if there is
        not already an integrand of the same name in the list.

        Args:
            integrand: The new Integrand object.

        Raises:
            IntegrandError: An error occurred during addition of the integrand.
        """
        affected_variable = integrand.affected_variable
        if affected_variable not in self.integrands:
            # add the new integrand name into the dictionary
            self.integrands[affected_variable] = integrand
            # add the integrand's affected variable into the integrator's
            # variable dictionary
            self._add_input_variables(integrand.input_variables)
            self._add_affected_variable(integrand.affected_variable)
        else:
            expression = "Duplicate key"
            message = ("Variable '{0}' is already affected by another "
                       "integrand in the integrand dictionary")
            raise (
                IntegrandError(expression, message.format(affected_variable)))

    def add_integrands(self, *integrands):
        """Adds multiple integrands to the integrand list

        Calls add_integrand() on each Integrand in a tuple

        Args:
            integrands: A tuple of Integrand objects
        """
        for integrand in integrands:
            self.add_integrand(integrand)

    def add_effector(self, effector):
        self.effectors.append(effector)
        self._add_input_variables(effector.input_variables)
        self._add_affected_variable(effector.affected_variable)

    def add_calculator(self, calculator):
        self.calculators.append(calculator)
        self._add_input_variables(calculator.input_variables)
        self._add_affected_variable(calculator.affected_variable)

    def add_calculators(self, *calculators):
        for calculator in calculators:
            self.add_calculator(calculator)

    def _add_input_variables(self, variables):
        for variable in variables:
            if variable not in self.input_variables:
                self.input_variables.append(variable)

    def _add_affected_variable(self, variable):
        if variable not in self.affected_variables:
            self.affected_variables.append(variable)

    def add_effectors(self, *effectors):
        for effector in effectors:
            self.add_effector(effector)

    def integrate(self, t_max, dt, ini_values=None):
        if ini_values is not None:
            self.ini_values = ini_values
        self.t_max = t_max
        self.dt = dt
        self.results["time"], self.n_steps = self._get_time_steps()
        self._preallocate_result_arrays()
        self._initialize_at_zero()
        for i in range(self.n_steps):
            if i > 0:
                self._integrate_to_step(i)
                self._apply_calculators(i)
                self._apply_effectors(i)
            else:
                self._apply_calculators(i)
                self._apply_effectors(i)

    def _apply_calculators(self, step):
        for calculator in self.calculators:
            input_variables = self._get_input_variables(calculator, step)
            affected_variable = calculator.affected_variable
            # directly change the value in the result vector
            # this is a difference to the effectors and enables you to
            # calculate several calculations which depend on each other
            # one after another
            new_value = calculator.function(*input_variables)
            self.results[affected_variable][step] = new_value

    def _apply_effectors(self, step):
        temp_result_dict = {}
        for effector in self.effectors:
            if effector.active:
                if effector.time >= self.results["time"][
                        step] or effector.time == -1:
                    # get effector input variables
                    input_variables = self._get_input_variables(effector, step)
                    affected_variable = effector.affected_variable
                    # store result of the effector calculation in a temporary
                    # dictionary with values labeled with variable names
                    temp_result_dict[affected_variable] = effector.function(
                        *input_variables)

                if effector.time != -1:
                    # deactivate used effector
                    effector.active = False

        # replace values in result arrays with ones from the temporary
        # dictionary
        for variable, value in temp_result_dict.items():
            self.results[variable][step] = value

    def _get_input_variables(self, carrier, step):
        input_variables = tuple(
            # get variables from the result array from index step for
            # variables with a result array, for the constants
            # always take the first and only one
            (self.results[variable][
                step] if variable in self.affected_variables or variable ==
                "time" else
             self.results[variable][0]) for variable in carrier.input_variables)
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
        for variable in self.affected_variables:
            # exclude the time array from pre-allocation, the time array
            # already exists at this point
            if variable != "time":
                self.results[variable] = np.zeros(self.n_steps)

    def _initialize_at_zero(self):
        for variable, value in self.ini_values.items():
            # only initialize variables that are used as inputs for
            # integrands or effectors
            if variable in self.affected_variables:
                self.results[variable][0] = value
            elif variable in self.input_variables:
                self.results[variable] = np.array([value])


class Integrand(object):
    """Object passed to an EulerIntegrator to integrate a function.

    The variable changed by an Integrand is calculated for the current time step
    using the variable value from the last time step.

    Attributes:
        function: Function handle of the function used to integrate a variable.
        affected_variable: A string containing the name of the variable being
            affected by function.
        input_variables: A list of strings containing the names of all input
            variables function takes, in the correct order from the function
            definition.
    """

    def __init__(self, func, affected_variable, input_variables):
        self.function = func
        self.affected_variable = affected_variable
        self.input_variables = input_variables


class Effector(object):
    """Object passed to an EulerIntegrator to change a variable.

    The variable changed by an Effector is changed in each time step after
    calculation of new variable values by first Integrands and then Calculators.
    An Effector should be used to evaluate conditions on the variables, such
    as clamping conditions, thresholds, etc.

    Attributes:
        function: A function handle of the function used to change a variable.
        affected_variable: A string containing the name of the variable
            being affected by function.
        input_variables: A list of strings containing the names of all input
            variables function takes, in the correct order from the function
            definition.
        time: A number specifying the time point at which the Effector should
            be applied. If time == -1, the Effector is applied at every time
            step.
    """

    def __init__(self, func, affected_variable, input_variables, time):
        self.function = func
        self.affected_variable = affected_variable
        self.input_variables = input_variables
        self.time = time
        self.active = True


class Calculator(object):
    """Object passed to an EulerIntegrator to calculate a variable.

    The variable changed by a Calculator is changed in each time step after
    calculation of new variable values by Integrands. It should be used to
    calculate variables that result from different integrated variables and
    therefore change, but are also needed as input for Integrators in each
    time step.

    Attributes:
        function: Function handle of the function used to calculate a variable.
        affected_variable: A string containing the name of the variable
            being affected by function.
        input_variables: A list of strings containing the names of all input
            variables function takes, in the correct order from the function
            definition.
    """

    def __init__(self, func, affected_variable, input_variables):
        self.function = func
        self.affected_variable = affected_variable
        self.input_variables = input_variables


class IntegrandError(Exception):
    """Exception raised for errors with integrands

    Attributes:
        expression: Input expression in which the error occurred
        message: Explanation of the error
    """

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message
