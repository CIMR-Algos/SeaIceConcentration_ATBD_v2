from __future__ import print_function

import numpy.ma as ma
import datetime
import re

fvals = dict()
fvals['f4']  = '-999'
fvals['int'] = '-999'

def readdata(filenames, dtypelist, sep=None, max_n_lines=None):
    """
    # Reads in dataset from file with known dtypes
    # Uses dtype list on form:
    #    [('column1 name', column1 dtype), ('col2 name', col2 dtype), etc]
    """

    if not isinstance(filenames,type([])):
        filenames = [filenames]

    loc_dtypelist  = []
    dt_convertions = []
    dt_formats     = []
    for i,ndt in enumerate(dtypelist):
        dt = ndt[1]
        if isinstance(dt,str) and ndt[1].startswith('DT/'):
            dt_convertions.append(i)
            dt_formats.append(ndt[1][3:])
            loc_dtypelist.append((ndt[0],datetime.datetime))
        else:
            loc_dtypelist.append(tuple(ndt))

    # We already know the file structure and can simply read the file line
    # by line, splitting them and adding them to a list for numpy.asarray.
    # Much faster and uses less memory than numpy.genfromtxt.

    templist = []

    for filename in filenames:
        with open(filename, 'r') as file:
            cl = 0
            for line in file:
                # update line counter (commented lines like header lines are counted)
                cl += 1
                #print "Line {}".format(cl,)
                # Skip any commented line, ie starts with #
                if line[0]=='#':
                    continue


                if max_n_lines is not None and cl >= max_n_lines:
                    break

                # numpy.asarray expects a tuple for each row when creating
                # a structured array, so we do that (to avoid a TypeError).
                try:
                    lin = line.strip('\n').split(sep)
                    if len(lin) != len(dtypelist):
                        print("Warning: found invalid line length in {}:{}: skip it (read {} and expected {})".format(filename,
                                                                                                               cl,len(lin),len(dtypelist)))
                        continue

                    for i,e in enumerate(lin):
                        if 'noval' in e or '-inf' in e:
                            dtype = dtypelist[i][-1]
                            if dtype in fvals.keys():
                                lin[i]=fvals[dtype]
                    for i,dtc in enumerate(dt_convertions):
                        try:
                            #print "DateTime Conversion : {} {}".format(lin[dtc],dt_formats[i],)
                            lin[dtc] = datetime.datetime.strptime(lin[dtc],dt_formats[i])
                        except ValueError as ve:
                            raise(ve)

                    templist.append(tuple(lin))
                except Exception as e:
                    raise ValueError('Error on %s:%d: (%s)' % (filename,cl,e))


    # We convert our list of tuples to a numpy array with correct dtypes.
    try:
        dataset = ma.asarray(templist, dtype=loc_dtypelist)
    except TypeError as e:
        # dtype columns doesn't match file columns, either wrong types (eg trying
        # to float() a string?) or wrong number of columns (eg missing data?).
        raise TypeError("There is a type error: {}".format( e ) )
    except Exception as e:
        raise(e)

    return dataset
