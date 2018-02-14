import qt
import time
import msvcrt
import os
import h5py
import sys
import numpy as np

from analysis.lib.fitting import fit, common
reload(common)

mos=qt.instruments['master_of_space']
green_aom=qt.instruments['GreenAOM']

def get_signal_to_noise():

    opt_ins = qt.instruments['opt1d_counts']

    qt.msleep(0.3)
    x,y = opt_ins.run(dimension='y', scan_length=1., nr_of_points=31, pixel_time=50, return_data=True, gaussian_fit=True)
    qt.msleep(0.2)
    fitargs= (np.min(y), np.max(y), x[np.argmax(y)], 0.1)
            #print fitargs, len(p)
    gaussian_fit = fit.fit1d(x, y,common.fit_gauss_pos, *fitargs, do_print=False,ret=True)


    if gaussian_fit['success']:

        p0=  gaussian_fit['params_dict']
        # qt.plot(x,y,name='sn_measurement',clear=True)
        # xp=linspace(min(x),max(x),100)
        # qt.plot(xp,gaussian_fit['fitfunc'](xp), name='sn_measurement')

        print 'signal/noise : {:.0f}/{:.0f}  = {:.2f}'.format(p0['A'],p0['a'],p0['A']/p0['a'])
        SN =p0['A']/p0['a']
    else:
        SN =0
    return SN


def optimize_z(z_range=0.4,nr_pts=5,green_power = 100e-6):
    z_current = mos.get_z()
    zs = np.linspace(z_current-z_range/2.,z_current+z_range/2.,nr_pts)
    SN = np.zeros(nr_pts)

    green_aom.set_power(green_power)
    for i,z in enumerate(zs):
        print i,z
        mos.set_z(z)
        SN[i] = get_signal_to_noise()
        if (msvcrt.kbhit() and msvcrt.getch()=='q'): 
            break   
    
    d = qt.Data(name='z_optimization_sn')
    d.add_coordinate('z (um)')
    d.add_value('SN')
    d.create_file()
    filename=d.get_filepath()[:-4]
    d.add_data_point(zs, SN)
    d.close_file()
    print filename
    fitargs= (np.min(SN), np.max(SN), zs[np.argmax(SN)], 0.1)

    gaussian_fit_SN = fit.fit1d(zs, SN,common.fit_gauss_pos, *fitargs, do_print=False,ret=True)


    
    p0=  gaussian_fit_SN['params_dict']
    
    #qt.plot(zs,SN,name='sn_measurement_z',clear=True)
    

    zp=np.linspace(min(zs),max(zs),100)
    
    # p_c = qt.plot(zp,gaussian_fit_SN['fitfunc'](zp), name='SN_vs_z')
    # p_c.save_png(filename+'.png')

    # p_c = qt.Plot2D(d, 'bO-', coorddim=0, name='z_optimization_sn', valdim=1, clear=True)
    # p_c.save_png(filename+'.png')


    print p0
    if gaussian_fit_SN['success']:

        print 'z0', p0['x0'] 

        if (p0['x0'] < min(zs)) or (p0['x0'] > max(zs)):
            'optimum out of scan range. setting to max point: ', zs[np.argmax(SN)]
            mos.set_z(zs[np.argmax(SN)])
        else:
            'optimize succeeded. new z:', p0['x0']
            mos.set_z(p0['x0'])

        #return gaussian_fit_SN
    else:
        'fit failed, setting back to intial positon', z_current
        mos.set_z(z_current)



    