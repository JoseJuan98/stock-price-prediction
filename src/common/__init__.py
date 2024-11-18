# -*- utf-8 -*-
"""Common module."""
from common.log import get_logger, msg_task
from common.stats import remove_outliers_iqr, check_stationarity, calculate_vif
from common.util import get_dataset, save_plot
