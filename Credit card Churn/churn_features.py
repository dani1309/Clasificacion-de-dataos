import numpy as np


def unknown_to_nan(data):
    return data.mask(data == 'Unknown', np.nan)


def add_engineered_features(data):
    d = data.copy()
    d['Avg_Ticket_Size'] = d['Total_Trans_Amt'] / np.maximum(d['Total_Trans_Ct'], 1)
    d['Inactivity_Contact_Ratio'] = d['Months_Inactive_12_mon'] * d['Contacts_Count_12_mon']
    d['Products_Per_Year'] = d['Total_Relationship_Count'] / (d['Months_on_book'] / 12.0)
    return d
