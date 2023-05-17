IN_PATH = 'D:/py_projects/earthquake_damage/data/'
OUT_PATH = 'D:/py_projects/earthquake_damage/output/'

# CAT_COLS = ['has_superstructure_adobe_mud', 'has_secondary_use_institution', 'has_secondary_use_industry',
#             'has_secondary_use_gov_office', 'ground_floor_type', 'has_secondary_use_use_police',
#             'has_superstructure_cement_mortar_stone', 'has_secondary_use_health_post', 'legal_ownership_status',
#             'has_superstructure_bamboo', 'other_floor_type', 'position', 'has_secondary_use_hotel',
#             'has_superstructure_mud_mortar_stone', 'has_superstructure_rc_non_engineered', 'has_superstructure_other',
#             'has_secondary_use_school', 'has_superstructure_stone_flag', 'has_superstructure_rc_engineered',
#             'foundation_type', 'plan_configuration', 'roof_type', 'has_secondary_use_other',
#             'has_secondary_use_agriculture', 'has_superstructure_timber', 'count_families',
#             'has_superstructure_cement_mortar_brick', 'has_secondary_use_rental', 'land_surface_condition',
#             'has_secondary_use', 'has_superstructure_mud_mortar_brick']
NUM_COLS = ['count_floors_pre_eq', 'age', 'area_percentage', 'height_percentage']
GEO_LVLS = ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id']

BINARY_COLS = ['has_secondary_use', 'has_secondary_use_agriculture',
               'has_secondary_use_gov_office', 'has_secondary_use_health_post', 'has_secondary_use_hotel',
               'has_secondary_use_industry', 'has_secondary_use_institution', 'has_secondary_use_other',
               'has_secondary_use_rental',
               'has_secondary_use_school', 'has_secondary_use_use_police', 'has_superstructure_adobe_mud',
               'has_superstructure_bamboo', 'has_superstructure_cement_mortar_brick',
               'has_superstructure_cement_mortar_stone',
               'has_superstructure_mud_mortar_brick', 'has_superstructure_mud_mortar_stone', 'has_superstructure_other',
               'has_superstructure_rc_engineered', 'has_superstructure_rc_non_engineered',
               'has_superstructure_stone_flag',
               'has_superstructure_timber']
MULTIARY_COLS = ['count_families', 'foundation_type', 'ground_floor_type', 'land_surface_condition',
                 'legal_ownership_status', 'other_floor_type',
                 'plan_configuration', 'position', 'roof_type']
