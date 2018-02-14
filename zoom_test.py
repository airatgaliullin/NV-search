
import qt
import time

import numpy as np
import msvcrt
import time


from scan import scan
import hdf5_data as h5
from analysis.lib.tools import toolbox as tb
import analysis.scripts.Fabrication.Display_scan2d as ds
import analysis.scripts.Fabrication.testCV2 as testcv
import measurement.lib.measurement2.measurement as m2
#import analysis.scripts.Fabrication.neural_image_recognition as nir



import cv2
import os


from measurement.lib.config import optimiz0rs as optcfg
reload(optcfg)

from measurement.scripts.lt3_scripts.basic import esr_NV_search
from measurement.scripts.lt3_scripts.basic import optimize_SN
from measurement.instruments.optimiz0r import optimiz0r as NV_search_optimizer

reload(optimize_SN)

#reload(nir)

reload(testcv)
reload(ds)
reload(esr_NV_search)

class NV_search:

	def zoom(self,x_scan,y_scan,z_plane,**kw):

		# if (i==0) and (j==0):
		# 	sleep_time=6
		# else:
		# 	sleep_time=2

		stop_scan = False



  		xstart=x_scan[0]
  		xstop=x_scan[1]
  		ystart=y_scan[0]
  		ystop=y_scan[1]
  		xsteps=abs(xstop-xstart)*10+1
  		ysteps=abs(ystop-ystart)*10+1
  		pixeltime=10.



		scan2d.set_x_position(xstart)
		scan2d.set_y_position(ystart)
		qt.instruments['GreenAOM'].turn_on()

		scan2d.set_xstart(xstart)
		scan2d.set_ystart(ystart)
		scan2d.set_xstop(xstop)
		scan2d.set_ystop(ystop)
		master_of_space.set_z(z_plane)
		


		scan2d.set_xsteps(xsteps)
		scan2d.set_ysteps(ysteps)
		scan2d.set_pixel_time(pixeltime)

		# while (scan2d.get_x_position()!=xstart) or (scan2d.get_y_position()!=ystart):
		# 	if (msvcrt.kbhit() and (msvcrt.getch() == 'q')): break
		# 	print scan2d.get_x_position()
		# 	pass
		
		qt.msleep(10)
		scan2d.set_is_running(True)


		while(scan2d.get_is_running()):
			qt.msleep(0.1)
			if (msvcrt.kbhit() and (msvcrt.getch() == 'q')): stop_scan=True
			if stop_scan: break

		timestamp2,folder = tb.latest_data(contains='scan2d', older_than = None,return_timestamp=True)


	def get_range(self,x_o,y_o,zoom,scan_size=2):
		n_of_pix=int(zoom/scan_size)
		coord_x=np.ones(n_of_pix)
		coord_y=np.ones(n_of_pix)

		for i in xrange(np.size(coord_x)):
			coord_x[i]=x_c-zoom/2+scan_size/2+i*scan_size

		for i in xrange(np.size(coord_y)):
			coord_y[i]=y_c-zoom/2+scan_size/2+i*scan_size
		return coord_x,coord_y



	def scan_area(self,z,search_range,x_scan,y_scan):
		#z_range=[-1.5]
		#x_range,y_range=self.get_range(x_o,y_o,zoom)

		oldest_folder=None
		blobs_list=[]
		xs_blobs=np.array([])
		ys_blobs=np.array([])
		blobs_counter=0
	
		ip=testcv.Image_processor()
		ip.config(search_range)
#		for k,z in enumerate(z_range):
		img_buf_total=[]
		

		folder=self.zoom(x_scan,y_scan,z)

		timestamp2,folder = tb.latest_data(contains='scan2d', older_than = None,return_timestamp=True)
		d= ds.DisplayScanFlim(folder)
		d.get_data()
		d.split_image(folder,x_scan,y_scan,save=True,colormap='gist_earth',search_range=search_range)

		for filename in os.listdir(folder):
			if 'x_c' in filename:
				cropped_image=ip.crop_image(folder, filename)
				x_blobs,y_blobs=ip.find_blobs(folder,filename,cropped_image,search_range,z)

				# xs_blobs = np.append(xs_blobs,x_blobs)
				# ys_blobs= np.append(ys_blobs,y_blobs)
				

				blobs_counter+=len(x_blobs)


				xy_blobs=np.append([x_blobs],[y_blobs],axis=0)
				xy_blobs=np.transpose(xy_blobs)
				blobs_list.append(xy_blobs)

		# blobs_list=np.append([xs_blobs],[ys_blobs],axis=0)

		return blobs_list,blobs_counter




	def optimize_blobs_position(self,blobs_list,z):
		e=esr_NV_search.esr_on_blob()
		

		local_optimiz0r=NV_search_optimizer('local_optimiz0r',dimension_set='lt3_NV_search')
		esr_to_check_coordinates=[]
		nv_coordinates=[]

		for i in xrange(len(blobs_list)):
			
			for j in xrange(len(blobs_list[i])):


				scan2d.set_x_position(blobs_list[i][j][0])
				scan2d.set_y_position(blobs_list[i][j][1])
				master_of_space.set_z(z)
				qt.msleep(5)
				continue_optimization=False
				success_last=False
				print 'x_search=', blobs_list[i][j][0]
				print 'y_search=',blobs_list[i][j][1]
				

				success_first=optimiz0r.optimize(cycles=1, dims=['y','x'])
				success_first=optimiz0r.optimize(cycles=1, dims=['y','x'])



				if (success_first):
					sigma=opt1d_counts.get_fit_result()[2]
					sigma_error=opt1d_counts.get_fit_error()[2]
					if (sigma>0.08) and (sigma<0.3) and (sigma_error<0.02):
						continue_optimization=True

					else:
						scan2d.set_x_position(blobs_list[i][j][0])
						scan2d.set_y_position(blobs_list[i][j][1])
						qt.msleep(2)
						success_first=optimiz0r.optimize(cycles=2, dims=['x','y'])

				else:
					scan2d.set_x_position(blobs_list[i][j][0])
					scan2d.set_y_position(blobs_list[i][j][1])
					qt.msleep(2)
					success_first=optimiz0r.optimize(cycles=2, dims=['x','y'])

				if (success_first):
					sigma=opt1d_counts.get_fit_result()[2]
					sigma_error=opt1d_counts.get_fit_error()[2]
					if (sigma>0.08) and (sigma<0.3) and (sigma_error<0.02):
						continue_optimization=True



				if (continue_optimization):
					optimize_SN.optimize_z()
					qt.msleep(1)
					success_last=optimiz0r.optimize(cycles=1,dims=['x','y'])

				if (success_last):
					print 'NV_coordinate_x=', scan2d.get_x_position()
					print 'NV_coordinate_y=', scan2d.get_y_position()
					esr_to_check_coordinates.append(([scan2d.get_x_position(),scan2d.get_y_position()]))
					name='_ESR_'+'x='+str(scan2d.get_x_position())[:4]+'_y='+str(scan2d.get_y_position())[:4]
					success=e.run(name=name)
					if (success):
						nv_coordinates.append(([scan2d.get_x_position(),scan2d.get_y_position()]))

		return esr_to_check_coordinates,nv_coordinates



	def manage_scans(self):
		z_range=[-3]
		x_o=0
		y_o=0
		search_range=2

		x_scan=[-5,5]
		y_scan=[-5,5]

		esr_check_coordinates=[]
		full_esr_to_check_coordinates=[]
		nv_coordinates=[]
		full_nv_coordinates=[]
		for k,z in enumerate(z_range):

######### Make scan_area to be compatible with new version of blob_list!!!!
			blobs_list,blobs_counter=self.scan_area(z,search_range, x_scan,y_scan)
			print 'Number of blobs', blobs_counter
		
######### Make esr compatible with new version of blob_list!!!!
		# 	esr_to_check_coordinates,nv_coordinates=self.optimize_blobs_position(blobs_list,z)
		# 	#print esr_check_coordinates
		# 	full_esr_to_check_coordinates.append(esr_to_check_coordinates)
		# 	full_nv_coordinates.append(nv_coordinates)


		# print 'full_esr_to_check_coordinates', full_esr_to_check_coordinates
		# print 'full_nv_coordinates', full_nv_coordinates
		


		#cv2.waitKey(0)
			





def start_scan():
	S=NV_search()
	S.manage_scans()









def start_scan_debug():
	blobs_list=[]
	xs_blobs=np.array([])
	ys_blobs=np.array([])
	filter_str=None
	folder=r'D:\measuring\data\20180208\194717_scan2d'
	search_range=2
	x_o=0
	y_o=0

	store=[]
	z=-3
	x_scan=[-12,12]
	y_scan=[-17,17]

	#dat = h5.HDF5Data(name='blobs_coordinates')
	ip=testcv.Image_processor()
	ip.config(search_range)
	S=NV_search()
	





	# I=nir.CNN()
	# I.train(r'D:\measuring\data\20180210\data\train',saved_file='trained_explorer_68',folder_validate=r'D:\measuring\data\20180210\data\validate')
	# folder_name=r'D:\measuring\data\20180211\130859_scan2d\images_to analyze\to_test'	
	# I.load_trained_model(load_file='trained_explorer_68')
	# I.analyze_images_from_folder(folder_name)
	# # e=esr_NV_search.esr_on_blob()
	# success=e.run(name='test_esr')
	# print success






	# d= ds.DisplayScanFlim(folder)
	# d.get_data()
	# d.split_image(folder,x_scan,y_scan,save=True,colormap='gist_earth',search_range=search_range)

	#filter_str=['x_c=-1 y_c=-3 ','x_c=-1 y_c=-10', 'x_c=-2 y_c=-1 ']
	filter_str=['x_c']
	qt.instruments['GreenAOM'].turn_on()

	for filename in os.listdir(folder):
		if any(s in filename for s in filter_str):
			if not 'search_range' in filename:

				
				
				cropped_image=ip.crop_image(folder, filename)
	 			x_blobs,y_blobs=ip.find_blobs(folder,filename,cropped_image,search_range,z)
				
				xs_blobs = np.append(xs_blobs,x_blobs)
				ys_blobs= np.append(ys_blobs,y_blobs)

				
				#xy_blobs=np.append([x_blobs],[y_blobs],axis=0)
				
				
	 			#xy_blobs=np.transpose(xy_blobs)
	
	blobs_list=np.append([xs_blobs],[ys_blobs],axis=0)
	
	print 'number of blobs',len(xs_blobs)

	del filter_str


	name='outcome_cordinates'
	dat = h5.HDF5Data(name=name)
	dat.create_dataset('z='+str(z)+' blobs_coordinates', data=np.transpose(blobs_list))
	m2.save_instrument_settings_file(dat)
	dat.close()
	
	# d = qt.Data(name=name)
	# d.add_coordinate('x')
	# d.add_value('y')
	# d.create_file()
	# d.add_data_point(blobs_counter[0][1], blobs_counter[0][1])
	# d.close_file()
				
	# esr_check_coordinates,nv_coordinates=S.optimize_blobs_position(blobs_list,z)
	# print 'esr_to_check', esr_check_coordinates
	# print 'nv_coordinates', nv_coordinates



	#cv2.imshow('img',img)


if __name__ == '__main__':
	debug=True
	if debug:
		start_scan_debug()
	else:
		start_scan()






	# 	for i,x_c in enumerate(x_range):
			
	# 		img_buf_vertical=[]
			
	# 		for j,y_c in enumerate(y_range):
	# 			self.zoom(i,j,x_0=x_c, y_0=y_c, z_plane=z, area_size=search_range)
	# 			timestamp2,folder = tb.latest_data(contains='scan2d', older_than = None,return_timestamp=True)
	# 			d= ds.DisplayScanFlim(folder)
	# 			d.get_data()
	# 			d.plot_data(folder,save=True,colormap='gist_earth')				

	# 			ip.config(scan_size=search_range)
	# 			cropped_image=ip.crop_image(folder)
	# 			x_blobs,y_blobs,img=ip.find_blobs(folder,cropped_image,zoom,x_c,y_c,z)
	# 			xy_blobs=np.append([x_blobs],[y_blobs],axis=0)
	# 			xy_blobs=np.transpose(xy_blobs)
	# 			blobs_list.append(xy_blobs)
	# 			img_buf_vertical.append(img)					
	# 		img_vertical_single_row=img_buf_vertical[0]
			
	# 		for q in xrange(1,len(img_buf_vertical)):
	# 			img_vertical_single_row=np.concatenate((img_buf_vertical[q],img_vertical_single_row),axis=0)

	# 		img_buf_total.append(img_vertical_single_row)
		
	# 	img_total=img_buf_total[0]
	# 	for q in xrange(1,len(img_buf_total)):
	# 		img_total=np.concatenate((img_total,img_buf_total[q]),axis=1)
		

	# 	cv2.imshow('total',img_total)
	# 	save_folder=folder+'_Total_scan'
	# #print save_folder
	# 	os.makedirs(save_folder)
	# 	fig_name='z='+str(z)+'x_c='+str(x_c)+'y_c='+str(y_c)+' Area='+str(zoom)+' um.png'

	# 	cv2.imwrite(os.path.join(save_folder,fig_name),img_total)

		#return blobs_list