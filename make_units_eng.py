"""
Make a unit_system consistent with usual OCC/FOT engineering units via P009.
"""

import asciitable
import pickle

dat = asciitable.read('/data/evans_i/IPCL/TDB/tdb_p009/tmsrment.txt',
                      Reader=asciitable.NoHeader, delimiter=",",
                      quotechar='"')

units_cxc = pickle.load(open('units_cxc.pkl'))

units_eng = dict((msid.upper(), unit) for msid, unit in zip(dat['col1'], dat['col5'])
                 if unit and msid.upper() in units_cxc)

pickle.dump(units_eng, open('units_eng.pkl', 'w'))
