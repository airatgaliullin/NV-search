import qt
import msvcrt
import numpy as np
from analysis.lib.fitting import fit, common,esr
from analysis.lib.tools import plot
from matplotlib import pyplot as plt

class esr_on_blob():
	def run(self,name, **kw):
		central_freq=2.88
		start_f = central_freq + 0.10#1.85#2.878 - 0.08 #   2.853 #2.85 #  #in GHz
		stop_f  = central_freq - 0.10#1.95#2.878 + 0.08 #   2.864 #2.905 #   #in GHz
		steps   = 101
		mw_power =kw.pop('mw_power',-18)#in dBm, never above -10
		green_power = kw.pop('green_power', 90e-6) #20e-6
		int_time = 150       #in ms
		reps = 2
		f_list = np.linspace(start_f*1e9, stop_f*1e9, steps)
		ins_smb = qt.instruments['SMB100']
		ins_adwin = qt.instruments['adwin']
		ins_counters = qt.instruments['counters']
		counter = 1
		MW_power = mw_power
		ins_counters.set_is_running(0)
		qt.mstart()
		ins_smb.set_power(MW_power)



    	
		
		
		
		

    # create data object
    	

    	
		ins_smb.set_iq('off')
		ins_smb.set_pulm('off')

		ins_smb.set_status('on')

		qt.msleep(0.2)
    #ins_counters.set_is_running(0)
		total_cnts = np.zeros(steps)
    	#qt.instruments['GreenAOM'].set_power(green_power)
		stop_scan=False
		for cur_rep in range(reps):
        
			print 'sweep %d/%d ...' % (cur_rep+1, reps)
        # optimiz0r.optimize(dims=['x','y','z'],int_time=50)
			for i,cur_f in enumerate(f_list):
				if (msvcrt.kbhit() and (msvcrt.getch() == 'q')): stop_scan=True
				ins_smb.set_frequency(cur_f)
            
				qt.msleep(0.02)
				total_cnts[i]+=ins_adwin.measure_counts(int_time)[counter-1]

				qt.msleep(0.01)

			p_c = qt.Plot2D(f_list, total_cnts, 'bO-', name='ESR', clear=True)

			if stop_scan: break



		ins_smb.set_status('off')

		d = qt.Data(name=name)
		d.add_coordinate('frequency [GHz]')
		d.add_value('counts')
		d.create_file()
		filename=d.get_filepath()[:-4]

		d.add_data_point(f_list, total_cnts)
		d.close_file()
		p_c = qt.Plot2D(d, 'bO-', coorddim=0, name='ESR', valdim=1, clear=True)
		p_c.save_png(filename+'.png')

		
		success=self.analyse_data(filename+'.dat')
		
		qt.mend()

		ins_counters.set_is_running(1)
		return success


    ##turn green off
    	#qt.instruments['GreenAOM'].set_power(0)

	def analyse_data(self,filename):
		data=np.loadtxt(filename)
		counts=data[:,1]
		freq=data[:,0]*1e-9
		guess_offset=counts[0]
		guess_amplitude=np.max(counts)-np.min(counts)
		guess_width=0.005
		guess_ctr=2.88
		success=False
		fit_result=fit.fit1d(freq,counts,esr.fit_ESR_gauss,guess_offset,guess_amplitude, guess_width,guess_ctr,
			do_print=False,ret=True)
		if fit_result['success']==1 and abs(fit_result['params_dict']['x0']-guess_ctr)<0.1:
			success=True
		plot.plot_fit1d(fit_result,np.linspace(np.min(freq),np.max(freq),1000),plot_data=True,add_txt =True, plot_title=None)
		plt.savefig(filename+'_fit'+'.png')
		plt.close()



		return success
