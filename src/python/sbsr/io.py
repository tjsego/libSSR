# todo: add YAML implementation using PyYAML

import numpy as np
from xml.etree import ElementTree

from .data import SupportingData, verify_data

######################
# Example XML format #
######################
# <sbsr level="0" version="0">
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
# </sbsr>
######################


def to_xml(inst: SupportingData):
    """
    Export a supporting data instance to XML

    :param inst: supporting data instance
    :return: XML root element
    """
    el_root = ElementTree.Element('sbsr',
                                  dict(level=str(inst.level),
                                       version=str(inst.version)))

    el_variable_names = ElementTree.SubElement(el_root,
                                               'variableNames')
    for name in inst.variable_names:
        el_name = ElementTree.SubElement(el_variable_names,
                                         'name')
        el_name.text = name

    el_simulation_times = ElementTree.SubElement(el_root,
                                                 'simulationTimes')
    el_simulation_times.text = ','.join([str(t) for t in inst.simulation_times])

    el_sample_size = ElementTree.SubElement(el_root,
                                            'sampleSize')
    el_sample_size.text = str(inst.sample_size)

    el_ecf_evals = ElementTree.SubElement(el_root,
                                          'ecfEvaluations')
    for i, sim_time in enumerate(inst.simulation_times):

        el_time = ElementTree.SubElement(el_ecf_evals,
                                         'time',
                                         dict(t=str(sim_time)))

        for j, name in enumerate(inst.variable_names):

            el_name = ElementTree.SubElement(el_time,
                                             'name',
                                             dict(n=name))
            vals = []
            for vt in inst.ecf_evals[i][j]:
                vals.extend([str(x) for x in vt])
            el_name.text = ','.join(vals)

    el_ecf_tval = ElementTree.SubElement(el_root,
                                         'ecfFinalTransformVarVals')
    for i, sim_time in enumerate(inst.simulation_times):

        el_time = ElementTree.SubElement(el_ecf_tval,
                                         'time',
                                         dict(t=str(sim_time)))

        for j, name in enumerate(inst.variable_names):

            el_name = ElementTree.SubElement(el_time,
                                             'name',
                                             dict(n=name))
            el_name.text = str(inst.ecf_tval[i, j])

    el_ecf_nval = ElementTree.SubElement(el_root,
                                         'ecfNumberOfEvaluations')
    el_ecf_nval.text = str(inst.ecf_nval)

    el_error_metric_mean = ElementTree.SubElement(el_root,
                                                  'errorMetricMean')
    el_error_metric_mean.text = str(inst.error_metric_mean)

    el_error_metric_stdev = ElementTree.SubElement(el_root,
                                                   'errorMetricStDev')
    el_error_metric_stdev.text = str(inst.error_metric_stdev)

    el_sig_figs = ElementTree.SubElement(el_root,
                                         'numberOfSigFigs')
    el_sig_figs.text = str(inst.sig_figs)

    return el_root


def from_xml(el_root: ElementTree.Element):
    """
    Load a supporting data instance from XML

    :param el_root: XML root element
    :return: supporting data instance
    """

    inst = SupportingData()
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

def to_json(inst: SupportingData):
    """
    Export a supporting data instance to JSON

    :param inst: supporting data instance
    :return: JSON data
    """

    ecf_evals = {}
    ecf_tval = {}
    for i, t in enumerate(inst.simulation_times):
        ecf_evals_t = {}
        ecf_tval_t = {}
        for j, n in enumerate(inst.variable_names):
            vals = []
            for vv in inst.ecf_evals[i][j]:
                for vt in vv:
                    vals.append(vt)
            ecf_evals_t[n] = vals
            ecf_tval_t[n] = inst.ecf_tval[i, j]
        ecf_evals[str(t)] = ecf_evals_t
        ecf_tval[str(t)] = ecf_tval_t

    json_data = dict(
        level=inst.level,
        version=inst.version,
        variableNames=inst.variable_names,
        simulationTimes=inst.simulation_times.tolist(),
        sampleSize=inst.sample_size,
        ecfEvaluations=ecf_evals,
        ecfFinalTransformVarVals=ecf_tval,
        ecfNumberOfEvaluations=inst.ecf_nval,
        errorMetricMean=inst.error_metric_mean,
        errorMetricStDev=inst.error_metric_stdev,
        numberOfSigFigs=inst.sig_figs
    )

    return json_data


def from_json(json_data: dict):
    """
    Load a supporting data instance from JSON

    :param json_data: JSON data
    :return: supporting data instance
    """

    inst = SupportingData()

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


#######################
# Example YAML format #
#######################
# level: 0
# version: 0
# variableNames: [S]
# simulationTimes: [0.0, 1.0]
# sampleSize: 1000
# ecfEvaluations:
#   - 0.0: {
#       S: [1.0, 0.0, 0.5, 0.5, 0.1, 0.1]
#   }
#   - 1.0: {
#       S: [1.0, 0.0, 0.2, 0.6, 0.2, 0.3]
#   }
# ecfFinalTransformVarVals:
#   - 0.0: {
#       S: 10.0
#   }
#   - 1.0: {
#       S: 20.0
#   }
# ecfNumberOfEvaluations: 3
# errorMetricMean: 0.001
# errorMetricStDev: 0.0005
# numberOfSigFigs: 6
#######################


def test_instance():
    inst = SupportingData()
    inst.variable_names = ['S']
    inst.simulation_times = np.asarray([0.0, 1.0], dtype=float)
    inst.sample_size = 1000
    inst.ecf_evals = np.asarray(
        [
            [
                [(1.0, 0.0), (0.5, 0.5), (0.1, 0.1)]
            ],
            [
                [(1.0, 0.0), (0.2, 0.6), (0.2, 0.3)]
            ]
        ],
        dtype=float
    )
    inst.ecf_tval = np.asarray(
        [
            [
                10.0
            ],
            [
                20.0
            ]
        ],
        dtype=float
    )
    inst.ecf_nval = 3
    inst.error_metric_mean = 0.001
    inst.error_metric_stdev = 0.0005
    inst.sig_figs = 6

    return verify_data(inst)


def test():
    from xml.dom import minidom

    inst = test_instance()

    el_xml = to_xml(inst)
    verify_data(from_xml(el_xml))
    dom = minidom.parseString(ElementTree.tostring(el_xml))
    print(dom.toprettyxml())

    data_json = to_json(inst)
    verify_data(from_json(data_json))
    print(data_json)


if __name__ == '__main__':
    test()
