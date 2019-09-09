import os
import six
import yaml
import warnings
import logging

logger = logging.getLogger('workflow')


class Strategy(object):

    def __init__(self):
        self.resource_pool = {}
        self.leaf_node = None
        self.leaf_out_file = None
        self.name = []
        self.config_file_path = ''

    # Getters
    def get_name(self):
        return self.name

    def get_leaf_properties(self):
        return self.leaf_node, self.leaf_out_file

    def get_resource_pool(self):
        return self.resource_pool

    def get_nodes_names(self):
        pieces = [n.split('_') for n in self.name]
        assert all(p[-1].isdigit() for p in pieces)
        return ['_'.join(p[:-1]) for p in pieces]

    def get_node_from_resource_pool(self, resource_key):
        try:
            return self.resource_pool[resource_key]
        except:
            logger.error('No node for output: %s', resource_key)
            raise

    # Setters
    def set_leaf_properties(self, node, out_file):
        self.leaf_node = node
        self.leaf_out_file = out_file

    def set_config_file(self, config_file_path):
        if not os.path.exists(config_file_path):
            raise ValueError(config_file_path + " file does not exist")
        self.config_file_path = config_file_path

    # Instance methods
    def append_name(self, name):
        self.name.append(name)

    def update_resource_pool(self, resources, override=False):
        for key, value in resources.items():
            if key in self.resource_pool and not override:
                raise Exception(
                    'Key %s already exists in resource pool, '
                    'replacing with %s ' % (key, value)
                )
            self.resource_pool[key] = value

    def __getitem__(self, resource_key):
        assert isinstance(resource_key, six.string_types)
        try:
            return self.resource_pool[resource_key]
        except:
            logger.error('No node for output: %s', resource_key)
            raise

    def __contains__(self, resource_key):
        assert isinstance(resource_key, six.string_types)
        return resource_key in self.resource_pool

    def fork(self):
        fork = Strategy()
        fork.resource_pool = dict(self.resource_pool)
        fork.leaf_node = self.leaf_node
        fork.out_file = str(self.leaf_out_file)
        fork.leaf_out_file = str(self.leaf_out_file)
        fork.name = list(self.name)
        return fork

    def update_data_config(self, rsc, config_file_path=''):
        if self.config_file_path == '' and config_file_path == '':
            raise ValueError("config_file_path is empty, impossible to write the resource")
        if config_file_path != '':
            self.set_config_file(config_file_path)

        try:
            with open(config_file_path, 'r') as sf:
                sublist = yaml.load(sf)
        except Exception as e:
            print("Subject list is not in proper YAML format. Please check "
                  "your file")
            raise e

    @staticmethod
    def get_forking_points(strategies):

        forking_points = []

        for strat in strategies:

            strat_node_names = set(strat.get_nodes_names())

            strat_forking = []
            for counter_strat in strategies:
                counter_strat_node_names = set(counter_strat.get_nodes_names())

                strat_forking += list(strat_node_names - counter_strat_node_names)

            strat_forking = list(set(strat_forking))
            forking_points += [strat_forking]

        return forking_points

    @staticmethod
    def get_forking_labels(strategies):

        fork_names = []

        # fork_points is a list of lists, each list containing node names of
        # nodes run in that strat/fork that are unique to that strat/fork
        fork_points = Strategy.get_forking_points(strategies)
        
        for fork_point in fork_points:
            
            fork_point.sort()

            fork_name = []

            for fork in fork_point:
                
                fork_label = ''

                if 'ants' in fork:
                    fork_label = 'ants'
                if 'fnirt' in fork:
                    fork_label = 'fnirt'
                elif 'flirt_register' in fork:
                    fork_label = 'linear-only'
                if 'afni' in fork:
                    fork_label = 'func-3dautomask'
                if 'fsl' in fork:
                    fork_label = 'func-bet'
                if 'fsl_afni' in fork:
                    fork_label = 'func-bet-3dautomask'    
                if 'epi_distcorr' in fork:
                    fork_label = 'dist-corr'
                if 'bbreg' in fork:
                    fork_label = 'bbreg'
                
                if 'aroma' in fork:
                    fork_label = 'aroma'
                if 'nuisance' in fork:
                    fork_label = 'nuisance'
                if 'frequency_filter' in fork:
                    fork_label = 'freq-filter'
                
                if 'median' in fork:
                    fork_label = 'median'
                if 'motion_stats' in fork:
                    fork_label = 'motion'
                if 'slice' in fork:
                    fork_label = 'slice'
                if 'anat_preproc_afni' in fork:
                    fork_label = 'anat-afni'
                if 'anat_preproc_bet' in fork:
                    fork_label = 'anat-bet'

                fork_name += [fork_label]

            fork_names.append('_'.join(set(fork_name)))

        return dict(zip(strategies, fork_names))
