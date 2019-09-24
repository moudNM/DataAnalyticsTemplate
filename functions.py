import pandas as pd
import numpy as np

"""
All the functions I created
"""

# write to file
def write_to_file(df, output_file_name, file_type):

    """
    different return types
    """
    def excel():
        writer = pd.ExcelWriter(output_file_name + '.xlsx')
        df.to_excel(writer, 'Sheet1', index=False)
        writer.save()

    def csv():
        df.to_csv(output_file_name + '.csv')

    switcher = {
        'excel': excel,
        'csv': csv
    }

    func = switcher.get(file_type)
    func()






