# todo: add YAML implementation using PyYAML
# todo: add support for optional parameter distribution specification using ProbOnto

import numpy as np
from typing import List, Optional
from xml.etree import ElementTree


######################
# Example XML format #
######################
# <ssr level="0" version="0">
#   <variableNames>
#       <name>S</name>
#   </variableNames>
#   <simulationTimes>0.0,1.0</simulationTimes>
#   <sampleSize>1000</sampleSize>
#   <ecfEvaluations>
#       <time t="0.0">
#           <name n="S">1.0,0.0,0.5,0.5,0.1,0.1</name>
#       </time>
#       <time "t"=1.0>
#           <name n="S">1.0,0.0,0.2,0.6,0.2,0.3</name>
#       </time>
#   </ecfEvaluations>
#   <ecfFinalTransformVarVals>
#       <time t="0.0">
#           <name n="S">10.0</name>
#       </time>
#       <time t="1.0">
#           <name n="S">20.0</name>
#       </time>
#   </ecfFinalTransformVarVals>
#   <ecfNumberOfEvaluations>3</ecfNumberOfEvaluations>
#   <errorMetricMean>0.001</errorMetricMean>
#   <errorMetricStDev>0.0005</errorMetricStDev>
#   <numberOfSigFigs>6</numberOfSigFigs>
# </ssr>
######################

#######################
# Example JSON format #
#######################
# {
#   "level": 0,
#   "version": 0,
#   "variableNames": ["S"],
#   "simulationTimes": [
#       0.0,
#       1.0
#   ],
#   "sampleSize": 1000,
#   "ecfEvaluations": {
#       "0.0": {
#           "S": [
#               1.0,
#               0.0,
#               0.5,
#               0.5,
#               0.1,
#               0.1
#           ]
#       },
#       "1.0": {
#           "S": [
#               1.0,
#               0.0,
#               0.2,
#               0.6,
#               0.2,
#               0.3
#           ]
#       }
#   },
#   "ecfFinalTransformVarVals": {
#       "0.0": {
#           "S": 10.0
#       },
#       "1.0": {
#           "S": 20.0
#       }
#   },
#   "ecfNumberOfEvaluations": 3,
#   "errorMetricMean": 0.001,
#   "errorMetricStDev": 0.0005,
#   "numberOfSigFigs": 6
# }
#######################


class SupportingData:
    """
    SSR supporting data
    """

    def __init__(self):

        self.level: int = 0
        """SSR level"""

        self.version: int = 0
        """SSR version"""

        self.variable_names: List[str] = []
        """Variable names"""

        self.simulation_times: np.ndarray = np.ndarray(())
        """Simulation times (1D float array)"""

        self.sample_size: int = 0
        """Size of sample"""

        self.ecf_evals: np.ndarray = np.ndarray(())
        """
        ECF evaluations
        
        - dim 0 is by simulation time
        - dim 1 is by variable name
        - dim 2 is by transform variable evaluation
        - dim 3 is real/imaginary
        """

        self.ecf_tval: np.ndarray = np.ndarray(())
        """
        ECF transform variable final value
        
        - dim 0 is by simulation time
        - dim 1 is by variable name
        """

        self.ecf_nval: int = 0
        """Number of ECF evaluations per time and name"""

        self.error_metric_mean: float = 0.0
        """Error metric mean, from test for reproducibility"""

        self.error_metric_stdev: float = 0.0
        """Error metric standard deviation, from test for reproducibility"""

        self.sig_figs: int = 0
        """Significant figures of sample data"""

    @classmethod
    def create(cls,
               variable_names: List[str],
               simulation_times: np.ndarray,
               sample_size: int,
               ecf_evals: np.ndarray,
               ecf_tval: np.ndarray,
               ecf_nval: int,
               error_metric_mean: float,
               error_metric_stdev: float,
               sig_figs: int):
        """
        Create an instance

        :param variable_names: Variable names
        :param simulation_times: Simulation times (1D float array)
        :param sample_size: Size of sample
        :param ecf_evals: ECF evaluations
            - dim 0 is by simulation time
            - dim 1 is by variable name
            - dim 2 is by transform variable evaluation
            - dim 3 is real/imaginary
        :param ecf_tval: ECF transform variable final value
            - dim 0 is by simulation time
            - dim 1 is by variable name
        :param ecf_nval: Number of ECF evaluations per time and name
        :param error_metric_mean: Error metric mean, from test for reproducibility
        :param error_metric_stdev: Error metric standard deviation, from test for reproducibility
        :param sig_figs: Significant figures of sample data
        :return: verified instance
        :raises ValueError: if any instance data is incomplete or incorrect
        """
        from libssr import __ssr_level__, __ssr_version__
        inst = SupportingData()
        inst.level = __ssr_level__
        inst.version = __ssr_version__
        inst.variable_names = variable_names
        inst.simulation_times = simulation_times
        inst.sample_size = sample_size
        inst.ecf_evals = ecf_evals
        inst.ecf_tval = ecf_tval
        inst.ecf_nval = ecf_nval
        inst.error_metric_mean = error_metric_mean
        inst.error_metric_stdev = error_metric_stdev
        inst.sig_figs = sig_figs
        return verify_data(inst)

    def to_xml(self):
        """
        Export to XML

        :return: XML root element
        """
        el_root = ElementTree.Element('ssr',
                                      dict(level=str(self.level),
                                           version=str(self.version)))

        el_variable_names = ElementTree.SubElement(el_root,
                                                   'variableNames')
        for name in self.variable_names:
            el_name = ElementTree.SubElement(el_variable_names,
                                             'name')
            el_name.text = name

        el_simulation_times = ElementTree.SubElement(el_root,
                                                     'simulationTimes')
        el_simulation_times.text = ','.join([str(t) for t in self.simulation_times])

        el_sample_size = ElementTree.SubElement(el_root,
                                                'sampleSize')
        el_sample_size.text = str(self.sample_size)

        el_ecf_evals = ElementTree.SubElement(el_root,
                                              'ecfEvaluations')
        for i, sim_time in enumerate(self.simulation_times):

            el_time = ElementTree.SubElement(el_ecf_evals,
                                             'time',
                                             dict(t=str(sim_time)))

            for j, name in enumerate(self.variable_names):

                el_name = ElementTree.SubElement(el_time,
                                                 'name',
                                                 dict(n=name))
                vals = []
                for vt in self.ecf_evals[i][j]:
                    vals.extend([str(x) for x in vt])
                el_name.text = ','.join(vals)

        el_ecf_tval = ElementTree.SubElement(el_root,
                                             'ecfFinalTransformVarVals')
        for i, sim_time in enumerate(self.simulation_times):

            el_time = ElementTree.SubElement(el_ecf_tval,
                                             'time',
                                             dict(t=str(sim_time)))

            for j, name in enumerate(self.variable_names):

                el_name = ElementTree.SubElement(el_time,
                                                 'name',
                                                 dict(n=name))
                el_name.text = str(self.ecf_tval[i, j])

        el_ecf_nval = ElementTree.SubElement(el_root,
                                             'ecfNumberOfEvaluations')
        el_ecf_nval.text = str(self.ecf_nval)

        el_error_metric_mean = ElementTree.SubElement(el_root,
                                                      'errorMetricMean')
        el_error_metric_mean.text = str(self.error_metric_mean)

        el_error_metric_stdev = ElementTree.SubElement(el_root,
                                                       'errorMetricStDev')
        el_error_metric_stdev.text = str(self.error_metric_stdev)

        el_sig_figs = ElementTree.SubElement(el_root,
                                             'numberOfSigFigs')
        el_sig_figs.text = str(self.sig_figs)

        return el_root

    @classmethod
    def from_xml(cls, el_root: ElementTree.Element):
        """
        Load an instance from XML

        :param el_root: XML root element
        :return: supporting data instance
        """

        inst = cls()
        inst.level = int(el_root.attrib['level'])
        inst.version = int(el_root.attrib['version'])

        for el_variable_names in el_root.findall('variableNames'):
            inst.variable_names.extend([el_name.text for el_name in el_variable_names.findall('name')])

        for el_simulation_times in el_root.findall('simulationTimes'):
            inst.simulation_times = np.asarray([float(v) for v in el_simulation_times.text.split(',')], dtype=float)

        for el_sample_size in el_root.findall('sampleSize'):
            inst.sample_size = int(el_sample_size.text)

        for el_ecf_nval in el_root.findall('ecfNumberOfEvaluations'):
            inst.ecf_nval = int(el_ecf_nval.text)

        for el_ecf_evals in el_root.findall('ecfEvaluations'):

            data = {}

            for el_time in el_ecf_evals.findall('time'):

                data_time = {}

                for el_name in el_time.findall('name'):

                    data_name = []

                    vals = [float(s) for s in el_name.text.split(',')]
                    for i in range(0, len(vals), 2):
                        data_name.append((vals[i], vals[i+1]))

                    data_time[el_name.attrib['n']] = data_name

                data[float(el_time.attrib['t'])] = data_time

            inst.ecf_evals = np.ndarray((len(inst.simulation_times), len(inst.variable_names), inst.ecf_nval, 2),
                                        dtype=float)
            for i, t in enumerate(inst.simulation_times):
                for j, n in enumerate(inst.variable_names):
                    for k in range(0, len(data[t][n]), 2):
                        inst.ecf_evals[i, j, k, :] = data[t][n][k]

        for el_ecf_tval in el_root.findall('ecfFinalTransformVarVals'):

            data = {}

            for el_time in el_ecf_tval.findall('time'):

                data_time = {}

                for el_name in el_time.findall('name'):

                    data_time[el_name.attrib['n']] = float(el_name.text)

                data[float(el_time.attrib['t'])] = data_time

            inst.ecf_tval = np.ndarray((len(inst.simulation_times), len(inst.variable_names)), dtype=float)
            for i, t in enumerate(inst.simulation_times):
                for j, n in enumerate(inst.variable_names):
                    inst.ecf_tval[i, j] = data[t][n]

        for el_error_metric_mean in el_root.findall('errorMetricMean'):
            inst.error_metric_mean = float(el_error_metric_mean.text)

        for el_error_metric_stdev in el_root.findall('errorMetricStDev'):
            inst.error_metric_stdev = float(el_error_metric_stdev.text)

        for el_sig_figs in el_root.findall('numberOfSigFigs'):
            inst.sig_figs = int(el_sig_figs.text)

        return inst

    def to_json(self):
        """
        Export to JSON

        :return: JSON data
        """

        ecf_evals = {}
        ecf_tval = {}
        for i, t in enumerate(self.simulation_times):
            ecf_evals_t = {}
            ecf_tval_t = {}
            for j, n in enumerate(self.variable_names):
                vals = []
                for vv in self.ecf_evals[i][j]:
                    for vt in vv:
                        vals.append(vt)
                ecf_evals_t[n] = vals
                ecf_tval_t[n] = self.ecf_tval[i, j]
            ecf_evals[str(t)] = ecf_evals_t
            ecf_tval[str(t)] = ecf_tval_t

        json_data = dict(
            level=self.level,
            version=self.version,
            variableNames=self.variable_names,
            simulationTimes=self.simulation_times.tolist(),
            sampleSize=self.sample_size,
            ecfEvaluations=ecf_evals,
            ecfFinalTransformVarVals=ecf_tval,
            ecfNumberOfEvaluations=self.ecf_nval,
            errorMetricMean=self.error_metric_mean,
            errorMetricStDev=self.error_metric_stdev,
            numberOfSigFigs=self.sig_figs
        )

        return json_data

    @classmethod
    def from_json(cls, json_data: dict):
        """
        Load an instance from JSON

        :param json_data: JSON data
        :return: supporting data instance
        """

        inst = cls()

        inst.level = json_data['level']
        inst.version = json_data['version']
        inst.variable_names = json_data['variableNames']
        inst.simulation_times = np.asarray(json_data['simulationTimes'], dtype=float)
        inst.sample_size = json_data['sampleSize']
        inst.ecf_nval = json_data['ecfNumberOfEvaluations']

        ecf_evals = json_data['ecfEvaluations']
        ecf_tval = json_data['ecfFinalTransformVarVals']
        inst.ecf_evals = np.ndarray((len(inst.simulation_times), len(inst.variable_names), inst.ecf_nval, 2), dtype=float)
        inst.ecf_tval = np.ndarray((len(inst.simulation_times), len(inst.variable_names)), dtype=float)
        for i, t in enumerate(inst.simulation_times):
            for j, n in enumerate(inst.variable_names):
                for k in range(inst.ecf_nval):
                    inst.ecf_evals[i, j, k, :] = ecf_evals[str(t)][n][2*k:2*(k+1)]
                inst.ecf_tval[i, j] = ecf_tval[str(t)][n]

        inst.error_metric_mean = json_data['errorMetricMean']
        inst.error_metric_stdev = json_data['errorMetricStDev']
        inst.sig_figs = json_data['numberOfSigFigs']

        return inst

    def __reduce__(self):
        return SupportingData.from_json, (self.to_json(),)

    def verify(self) -> bool:
        """
        Verify completeness and accuracy of data.

        This is a soft test.

        :return: True if verified, otherwise False
        """

        try:
            verify_data(self)
            return True
        except ValueError:
            return False

    def error_info(self) -> Optional[str]:
        """
        Return error information, if any

        :return: Error information, if any. Otherwise None
        """
        try:
            verify_data(self)
            return None
        except ValueError as err:
            return str(err)


def verify_data(inst: SupportingData):
    """
    Verify an instance

    :param inst: an instance
    :return: the instance
    :raises ValueError: when any element of the instance is incomplete or incorrect.
    """
    if not inst.variable_names:
        raise ValueError('No variable names')
    if inst.sample_size <= 0:
        raise ValueError(f'Incorrect sample size ({inst.sample_size})')
    if inst.ecf_nval <= 0:
        raise ValueError(f'Incorrect number of ECF evaluations ({inst.ecf_nval})')
    if inst.sig_figs <= 0:
        raise ValueError(f'Incorrect number of significant figures ({inst.sig_figs})')
    if inst.ecf_evals.shape[0] != len(inst.simulation_times):
        raise ValueError(f'Incorrect ECF shape: times ({inst.ecf_evals.shape[0], len(inst.simulation_times)})')
    if inst.ecf_tval.shape[0] != len(inst.simulation_times):
        raise ValueError(
            f'Incorrect ECF transform variable shape: times ({inst.ecf_tval.shape[0], len(inst.simulation_times)})'
        )
    if inst.ecf_evals.shape[1] != len(inst.variable_names):
        raise ValueError(f'Incorrect ECF shape: names ({inst.ecf_evals.shape[1], len(inst.variable_names)})')
    if inst.ecf_tval.shape[1] != len(inst.variable_names):
        raise ValueError(
            f'Incorrect ECF transform variable shape: names ({inst.ecf_tval.shape[1], len(inst.variable_names)})'
        )
    if inst.ecf_evals.shape[2] != inst.ecf_nval:
        raise ValueError(f'Incorrect ECF shape: evaluations ({inst.ecf_evals.shape[2], inst.ecf_nval})')

    return inst


def test_instance() -> SupportingData:
    """Generate a test instance"""

    return SupportingData.create(
            variable_names=['S'],
            simulation_times=np.asarray([0.0, 1.0], dtype=float),
            sample_size=1000,
            ecf_evals=np.asarray(
                [
                    [
                        [(1.0, 0.0), (0.5, 0.5), (0.1, 0.1)]
                    ],
                    [
                        [(1.0, 0.0), (0.2, 0.6), (0.2, 0.3)]
                    ]
                ],
                dtype=float
            ),
            ecf_tval=np.asarray(
                [
                    [
                        10.0
                    ],
                    [
                        20.0
                    ]
                ],
                dtype=float
            ),
            ecf_nval=3,
            error_metric_mean=0.001,
            error_metric_stdev=0.0005,
            sig_figs=6
        )


def test():
    """Do testing"""
    from xml.dom import minidom

    inst = test_instance()

    el_xml = inst.to_xml()
    verify_data(SupportingData.from_xml(el_xml))
    dom = minidom.parseString(ElementTree.tostring(el_xml))
    print(dom.toprettyxml())

    data_json = inst.to_json()
    verify_data(SupportingData.from_json(data_json))
    print(data_json)


if __name__ == '__main__':
    test()
