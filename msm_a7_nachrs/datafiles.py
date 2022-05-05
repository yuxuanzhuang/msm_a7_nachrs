from pkg_resources import resource_filename

BGT = resource_filename(__name__,
                        'datafile/bgt.pdb')
EPJ = resource_filename(__name__,
                        'datafile/epj.pdb')
EPJPNU = resource_filename(__name__,
                           'datafile/epjpnu.pdb')

CLIMBER_BGT_EPJ = resource_filename(__name__,
                                    'datafile/climber/climber_bgt_epj.pdb')
CLIMBER_BGT_EPJPNU = resource_filename(__name__,
                                       'datafile/climber/climber_bgt_epjpnu.pdb')
CLIMBER_EPJ_BGT = resource_filename(__name__,
                                    'datafile/climber/climber_epj_bgt.pdb')
CLIMBER_EPJ_EPJPNU = resource_filename(__name__,
                                       'datafile/climber/climber_epj_epjpnu.pdb')
CLIMBER_EPJPNU_EPJ = resource_filename(__name__,
                                       'datafile/climber/climber_epjpnu_epj.pdb')
CLIMBER_EPJPNU_BGT = resource_filename(__name__,
                                       'datafile/climber/climber_epjpnu_bgt.pdb')

CLIMBER_BGT_EPJ_TRANSITION = resource_filename(__name__,
                                               'datafile/climber/climber_bgt_epj.xtc')
CLIMBER_BGT_EPJPNU_TRANSITION = resource_filename(__name__,
                                                  'datafile/climber/climber_bgt_epjpnu.xtc')
CLIMBER_EPJ_BGT_TRANSITION = resource_filename(__name__,
                                               'datafile/climber/climber_epj_bgt.xtc')
CLIMBER_EPJ_EPJPNU_TRANSITION = resource_filename(__name__,
                                                  'datafile/climber/climber_epj_epjpnu.xtc')
CLIMBER_EPJPNU_EPJ_TRANSITION = resource_filename(__name__,
                                                  'datafile/climber/climber_epjpnu_epj.xtc')
CLIMBER_EPJPNU_BGT_TRANSITION = resource_filename(__name__,
                                                  'datafile/climber/climber_epjpnu_bgt.xtc')
