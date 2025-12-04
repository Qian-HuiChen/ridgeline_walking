#!/usr/bin/env python
# coding: utf-8

#import package
import os,math
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import matplotlib.colors as mcolors
from scipy.optimize import curve_fit
from scipy import ndimage, signal
import warnings
from scipy.interpolate import UnivariateSpline
warnings.filterwarnings('ignore')
Pi = math.pi
font1 = {'family' : 'Times New Roman',
'weight' : 'black',
'size'   : 18}
font2 = {'family' : 'Times New Roman',
'weight' : 'black',
'size'   : 22}
plt.style.use('seaborn-poster')
plt.rcParams['axes.grid'] = 'False'
plt.rcParams['grid.alpha'] = 0.3


def get_ellipse(e_x, e_y, a, b, e_angle):                        #e_angle 跟x正半轴夹角
    angles_circle = np.arange(0, 2 * np.pi, 0.01)
    x = []
    y = []
    for angles in angles_circle:
        or_x = a * math.cos(angles)
        or_y = b * math.sin(angles)
        length_or = np.sqrt(or_x * or_x + or_y * or_y)
        or_theta = math.atan2(or_y, or_x)
        new_theta = or_theta + (e_angle)/180.*math.pi
        new_x = e_x + length_or * math.cos(new_theta)
        new_y = e_y + length_or * math.sin(new_theta)
        x.append(new_x)
        y.append(new_y)
    return x, y

def get_pa(gal_index):
    cat = open('/Users/chenqianhui/Downloads/galfit_catalog_updated.dat', 'r')
    q0 = 0.2  # for spiral galaxies
    for line in cat:
        if line.startswith(gal_index):
            data = line.split()
            re = float(data[1])
            n = float(data[2])
            q = float(data[3])
            pa = float(data[17]) + 90  # F160W, from x to y, so +90
            cos_i = np.sqrt((q * q - q0 * q0) / (1 - q0 * q0))  # q0 = 0.2, cos_i = cos(inclination)
            incl = math.degrees(math.acos(cos_i))  # inclination in degrees
            break
    cat.close()
    return re, pa, q, incl, n

def get_image(gal_index, re, pa, q):
    #Read in image and headers
    image_path = '/Users/chenqianhui/Downloads/2111_cutout/' + gal_index + '_JWST_NIRCAM_F115W_cutout_6.0_arcsec.GAB.fits'
    hdu = fits.open(image_path)
    ny, nx = hdu[0].data.shape
    crpix1 = hdu[0].header['CRPIX1']
    crpix2 = hdu[0].header['CRPIX2']
    pixscale = hdu[0].header['CUTPSOUT']
    re_inpix = re / pixscale  # re in pixels

    #cut out the image
    x_lim0 = math.floor(crpix1 - re_inpix*2)
    x_lim1 = math.ceil(crpix1 + re_inpix*2)
    y_lim0 = math.floor(crpix2 - re_inpix*2)
    y_lim1 = math.ceil(crpix2 + re_inpix*2)
    print('x_lim0:', x_lim0, 'x_lim1:', x_lim1, 'y_lim0:', y_lim0, 'y_lim1:', y_lim1)
    image_data = hdu[0].data[x_lim0:x_lim1, y_lim0:y_lim1]
    hdu.close()

    plt.figure(figsize=(6, 6))
    plt.imshow(image_data, norm = mcolors.PowerNorm(gamma=0.5, vmin=0.1, vmax=2), cmap='gray', origin='lower')
    plt.colorbar()

    xc = np.where(image_data == np.max(image_data))[1][0]  # x center
    yc = np.where(image_data == np.max(image_data))[0][0]  #
    z1 = get_ellipse(xc, yc, a = re_inpix/2, b = q * re_inpix/2 , e_angle=pa)
    plt.plot(z1[0],z1[1],alpha = 1, color = 'black',linestyle='dashed', linewidth = 1,label='0.5$\\times R_{e}$')
    plt.title(gal_index + ' JWST NIRCAM F115W', fontdict=font2)
    plt.show()
    return image_data, xc, yc, [x_lim0, nx-x_lim1, y_lim0, ny-y_lim1], re_inpix   

def get_dedist(image, xc, yc, pa, incl):
    #calculate deprojected distance（in unit of pixel）
    dedist = np.full(image.shape,np.nan)
    xcrot = xc * math.cos(pa/180.*Pi) + yc * math.sin(pa/180*Pi)
    ycrot = -xc * math.sin(pa/180.*Pi) + yc * math.cos(pa/180*Pi)
    xc = xcrot
    yc = ycrot/math.cos(incl/180.*Pi)
    for i in range(0,image.shape[0]):
        for j in range(0,image.shape[1]):
            xrot = j * math.cos(pa/180.*Pi) + i * math.sin(pa/180*Pi)
            yrot = -j * math.sin(pa/180.*Pi) + i * math.cos(pa/180*Pi)
            xdpj = xrot
            ydpj = yrot/math.cos(incl/180.*Pi)
            dedist[i][j] = np.sqrt(np.square(xdpj-xc) + np.square(ydpj-yc))
    return dedist


def cal_phi_map(im, xc, yc, pa, incl):
    """
    Calculate the azimuthal angle (phi) map for the image.
    :param im: 2D numpy array of the image data
    :param xc: x-coordinate of the center
    :param yc: y-coordinate of the center
    :param pa: position angle in degrees
    :param incl: inclination in degrees
    :return: 2D numpy array of azimuthal angles in degrees
    """
    #calculate phi0 - Northern
    xcrot = xc * math.cos(pa/180.*Pi) + yc * math.sin(pa/180*Pi)
    ycrot = -xc * math.sin(pa/180.*Pi) + yc * math.cos(pa/180*Pi)
    xc = xcrot
    yc = ycrot/math.cos(incl/180.*Pi)
    phi0_im = np.full(im.shape,np.nan)
    for i in range(0,im.shape[0]):
        for j in range(0,im.shape[1]):
            #if haspl[i][j] > -99:
            xrot = j * math.cos(pa/180.*Pi) + i * math.sin(pa/180*Pi)
            yrot = -j * math.sin(pa/180.*Pi) + i * math.cos(pa/180*Pi)
            xdpj = xrot
            ydpj = yrot/math.cos(incl/180.*Pi)
            if (ydpj > yc) and (xdpj > xc):
                phi0_im[i][j] = math.atan((ydpj-yc)/(xdpj-xc))/Pi*180. + 0 #in unit of degree
            if (ydpj > yc) and (xdpj < xc):
                phi0_im[i][j] = math.atan((ydpj-yc)/(xdpj-xc))/Pi*180. + 180. #in unit of degree
            if (ydpj < yc) and (xdpj < xc):
                phi0_im[i][j] = math.atan((ydpj-yc)/(xdpj-xc))/Pi*180. + 180. #in unit of degree
            if (ydpj < yc) and (xdpj > xc):
                phi0_im[i][j] = math.atan((ydpj-yc)/(xdpj-xc))/Pi*180. + 360.#in unit of degree
            if (ydpj == yc) and (xdpj > xc) :
                phi0_im[i][j] = 0.
            if (ydpj == yc) and (xdpj < xc) :
                phi0_im[i][j] = 180.
            if (ydpj < yc) and (xdpj == xc) :
                phi0_im[i][j] = 270.
            if (ydpj > yc) and (xdpj == xc) :
                phi0_im[i][j] = 90.
    return phi0_im


# Calculate mean for each radius bin (bin width = 2 pixels)
def get_mean_profile(image, bin_width=1, dedist = None):
    """Subtract the mean value of each radial bin."""
    max_radius = int(np.nanmax(dedist))
    radii = np.arange(0, max_radius, bin_width)
    mean_profile = []

    for r in radii:
        mask = (dedist >= r) & (dedist < r + bin_width)
        if np.any(mask):
            mean_val = np.nanmean(image[mask])
        else:
            mean_val = np.nan
        mean_profile.append(mean_val)

    # Plot the radial mean profile
    # plt.figure(figsize=(8, 6))
    # plt.plot(radii + bin_width/2, mean_profile, marker='o')
    # plt.xlabel('Radius (pixels)')
    # plt.ylabel('Mean value')
    # plt.title('Radial Mean Profile (bin width = 2 pixels)')
    # plt.show()
    
    im_submean = np.zeros_like(image)
    for r in range(len(mean_profile)):
        mask = (dedist >= r * bin_width) & (dedist < (r + 1) * bin_width)
        im_submean[mask] = image[mask] - mean_profile[r]

    # Plot the subtracted image
    # plt.figure(figsize=(8, 8))
    # plt.imshow(im_submean, norm=mcolors.PowerNorm(gamma=0.5, vmin=-0.1, vmax=0.6), cmap='gray', origin='lower')
    # plt.colorbar()
    # plt.title('Image after subtracting radial mean', fontdict=font2)
    # plt.show()
    return im_submean




def sersic_2d(x, y, xc, yc, re, n, q, pa, Ie):
    # pa in degrees, counterclockwise from x axis
    theta = np.deg2rad(pa)
    x_rot = (x - xc) * np.cos(theta) + (y - yc) * np.sin(theta)
    y_rot = -(x - xc) * np.sin(theta) + (y - yc) * np.cos(theta)
    r_ell = np.sqrt(x_rot**2 + (y_rot/q)**2)
    bn = 2 * n - 1/3  # Approximation for b_n
    return Ie * np.exp(-bn * ((r_ell / re)**(1/n) - 1))


def find_peak_cnt(phase_dgm, flx_len, promin1=100, promin2=10): 
    markout = 1 #the footprint of the ant will be marked by 1, 2, 3, ...
    #phase_dgm is the 2D array of the phase diagram, colour-coded by whatever you want.
    #flx_len is the width of the strip, which will be later send to signal.find_peaks to find the localised peak
    #You need to play around the flx_len as it differs at various dataset.
    #promin1, promin2, how significant you require your peak to be. If finding promin1 fails, the code will try a second time for promin2.
    
    flag = 1 #footprint of the ant will be flagged from 1 or any constant you want
    marker = np.full(phase_dgm.shape, np.nan) #define an empty footprint map
    hgt = phase_dgm.shape[0] #read out the height of the phase diagram
    strip = phase_dgm[int(hgt/2)-flx_len:int(hgt/2)+flx_len, :] #the strip around the middle radius
    start_point = (int(hgt/2)-flx_len + np.where(strip == np.nanmax(strip))[0][0], np.where(strip == np.nanmax(strip))[1][0]) #the position where I drop the ant, by finding the brightest pixel in the strip
    marker[start_point] = flag 
    for i in range(start_point[1]+1, phase_dgm.shape[1]): #ask the ant to walk from starting anchor toward INCREASING azimuth, ending at 360deg
        new_strip = phase_dgm[start_point[0]-flx_len:start_point[0]+flx_len, i] #get a new strip, taking regime of +/- flx_len in the R direction, and 1 grid in the azimuth direction
        try:
            strip_arch = signal.find_peaks(new_strip, prominence=promin1)[0] #find a localised peak in the new strip, the signal needs to be more prominant than promin1
            1/len(strip_arch[0])
        except:
            strip_arch = signal.find_peaks(new_strip, prominence=promin2)[0] #if promin1 fail, lower the threshold to promin2 (you can skip second search this if you want)
        if len(strip_arch) >= 1: #if there is a prominant peak found in the new strip
            start_point = (strip_arch[0]+start_point[0]-flx_len, i) #move the ant to the new peak
            flag += 1
            marker[start_point[0]][i] = flag #highlight the footprint as 1, 2, 3, ...
            
        if start_point[0]-flx_len < 0: #do nothing if there is no "good" peak found in the new strip
            break
    
    #As (R1, 360deg) is the same as (R1, 0deg), we want the ant to jump from the right end to the left end of the phase diagram
    for i in range(0, phase_dgm.shape[1]): #ask the ant to keep walking from 0 deg toward INCREASING azimuth, ending at 360deg
        new_strip = phase_dgm[start_point[0]-flx_len:start_point[0]+flx_len, i]
        try:
            strip_arch = signal.find_peaks(new_strip, prominence=promin1)[0]
            1/len(strip_arch[0])
        except:
            strip_arch = signal.find_peaks(new_strip, prominence=promin2)[0]
        if len(strip_arch) >= 1:
            start_point = (strip_arch[0]+start_point[0]-flx_len, i)
            marker[start_point[0]][i] = flag
            flag += 1
        if start_point[0]-flx_len < 0:
            break
            
    strip = phase_dgm[int(hgt/2)-flx_len:int(hgt/2)+flx_len, :]
    tmp = 0
    start_point = (int(hgt/2)-flx_len + np.where(strip == np.nanmax(strip))[0][0], np.where(strip == np.nanmax(strip))[1][0])
    tmp = start_point[0] #the above four lines are going back to the starting anchor
    flag = 0
    for i in range(start_point[1]-1, -1, -1): #ask the ant to walk from the starting anchor, towards decreasing azimuth, ending at 0deg
        new_strip = phase_dgm[start_point[0]-flx_len:start_point[0]+flx_len, i]#same as the loop above, but in the decreasing θ direction
        try:
            strip_arch = signal.find_peaks(new_strip, prominence=promin1)[0]
            1/len(strip_arch[0])
        except:
            strip_arch = signal.find_peaks(new_strip, prominence=promin2)[0]
        if len(strip_arch) >= 1:
            if strip_arch[0] < 2:
                flag += 1
            start_point = (strip_arch[0]+start_point[0]-flx_len, i)
            marker[start_point[0]][i] = markout
            markout += 1
        if start_point[0]-flx_len < 0:
            break
        
    #As (R2, 0deg) is the same pixel as (R2, 360 deg), the ant jumps from the left end to the right end of the phase diagram
    for i in range(phase_dgm.shape[1]-1, 0, -1): 
        new_strip = phase_dgm[start_point[0]-flx_len:start_point[0]+flx_len, i]
        try:
            strip_arch = signal.find_peaks(new_strip, prominence=promin1)[0]
            1/len(strip_arch[0])
        except:
            strip_arch = signal.find_peaks(new_strip, prominence=promin2)[0]
        if len(strip_arch) >= 1:
            if strip_arch[0] < 2:
                flag += 1
            start_point = (strip_arch[0]+start_point[0]-flx_len, i)
            marker[start_point[0]][i] = flag
            flag += 1
        if start_point[0]-flx_len < 0:
            break
    return(marker)



def img2phase(im_submean, dedist, phi0_im, num_r_bins=60+1, phi_bin_width = 4):
    # Define log-spaced radius bins (avoid log(0) by starting from a small positive value)
    r_min = np.nanmin(dedist[dedist > 0])
    r_max = np.nanmax(dedist)

    r_bins = np.logspace(np.log(r_min), np.log(r_max), num=num_r_bins, base=np.e)
    phi_max = 360
    phi_bins = np.arange(0, phi_max + phi_bin_width, phi_bin_width)

    # Prepare output array
    phase_dig = np.full((len(r_bins)-1, len(phi_bins)-1), np.nan)

    # Flatten arrays for easier indexing
    dedist_flat = dedist.flatten()
    phi0_flat = phi0_im.flatten()
    im_flat = im_submean.flatten()

    # Bin and calculate mean for each (r, phi) bin
    for i in range(len(r_bins)-1):
        for j in range(len(phi_bins)-1):
            mask = (
                (dedist_flat >= r_bins[i]) & (dedist_flat < r_bins[i+1]) &
                (phi0_flat >= phi_bins[j]) & (phi0_flat < phi_bins[j+1])
            )
            if np.any(mask):
                vals = im_flat[mask]
                vals = vals[~np.isnan(vals)]
                if len(vals) > 0:
                    phase_dig[i, j] = np.nanmean(vals)

    # Plot with log y-axis
    plt.figure(figsize=(16, 6))
    plt.imshow(
        phase_dig,
        aspect='auto',
        origin='lower',
        extent=[phi_bins[0], phi_bins[-1], np.log(r_bins[0]), np.log(r_bins[-1])],
        cmap='jet',
        norm=mcolors.PowerNorm(gamma=0.4, vmin=np.nanpercentile(phase_dig, 10), vmax=np.nanpercentile(phase_dig, 95))
    )
    plt.xlabel('Azimuthal Angle $\\Phi$ (deg)', font1)
    plt.ylabel('log$_e$(Radius)', font1)
    plt.yticks(
        [ np.log(10), np.log(20), np.log(40), np.log(80)],  # adjust step for readability
        labels=['log10', 'log20', 'log40', 'log80']  # adjust labels as needed
    )
    plt.ylim(np.log(10), np.log(r_bins[-1]))
    plt.colorbar(label='F115W mean subtracted')
    plt.title('Phase Diagram (log(r), φ)', font1)
    plt.show()
    return phase_dig, r_bins, phi_bins


def marker2spiral_map(img, marker1, marker2, r_bins, phi_bins, dedist, phi0_im):
    spmask = np.full(img.shape, np.nan)
    mask = np.where((marker1 > 0) | (marker2 > 0))
    for i in range(len(mask[0])):
        # Convert bin indices to physical values
        r_bin_idx = mask[0][i]
        phi_bin_idx = mask[1][i]
        # Get the radius and phi value for this bin (use bin centers)
        r_center = 0.5 * (r_bins[r_bin_idx] + r_bins[r_bin_idx + 1])
        phi_center = 0.5 * (phi_bins[phi_bin_idx] + phi_bins[phi_bin_idx + 1])
        # Create masks for pixels in this (r, phi) bin
        r_mask = np.abs(dedist - r_center) < (r_bins[r_bin_idx + 1] - r_bins[r_bin_idx]) / 2
        phi_mask = np.abs(phi0_im - phi_center) < (phi_bins[phi_bin_idx + 1] - phi_bins[phi_bin_idx]) / 2
        spmask[(r_mask) & (phi_mask)] = 1
    return spmask


def compare_img_spmask(img, im_submean, spmask, isSave = True):
    fig, axs = plt.subplots(1, 3, figsize=(17, 6))

    # Left panel: im_compile only
    im0 = axs[0].imshow(img, norm=mcolors.PowerNorm(gamma=0.5, vmin=0.1, vmax=2), cmap='gray', origin='lower')
    axs[0].set_title('F115W')
    cbar = plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
    cbar.set_ticks([0.1, 0.5, 1, 1.5, 2])

    # Right panel: im_compile with spmask overlay
    im1 = axs[1].imshow(im_submean, norm=mcolors.PowerNorm(gamma=0.5, vmin=-0.1, vmax=0.6), cmap='gray', origin='lower')
    axs[1].set_title('Subtracted Mean')
    cbar = plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
    cbar.set_ticks([-0.1, 0,  0.2, 0.6 ])

    # Right panel: im_compile with spmask overlay
    im2 = axs[2].imshow(img, norm=mcolors.PowerNorm(gamma=0.5, vmin=0.1, vmax=2), cmap='gray', origin='lower')
    axs[2].imshow(spmask, cmap='winter_r', origin='lower', vmax=1)
    axs[2].set_title('F115W + spmask')
    cbar = plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)
    cbar.set_ticks([0.1, 0.5, 1, 1.5, 2])

    if isSave:
        plt.tight_layout()
        plt.savefig('/Users/chenqianhui/Program/SpiralGal/MSA-3D/2111_spmask_compare.png', dpi=300, bbox_inches='tight')

    plt.show()

    return None


def complete_spmask(spmask, cutout_edge, isSave = True, file_name = "spmask" , gal_name='Unknown', save_dir = '/Users/chenqianhui/Program/SpiralGal/MSA-3D/'):
    """
    Complete the spiral mask to the original image size.
    """
    spmask_0 = np.vstack([np.full([cutout_edge[2], spmask.shape[1]], np.nan), spmask])
    spmask_1 = np.vstack([spmask_0, np.full([cutout_edge[3], spmask.shape[1]], np.nan)])
    spmask_2 = np.hstack([np.full([spmask_1.shape[0], cutout_edge[0]], np.nan), spmask_1])
    spmask_final = np.hstack([spmask_2, np.full([spmask_2.shape[0], cutout_edge[1]], np.nan)])

    if isSave:
        fits.writeto(save_dir+ gal_name+'_'+file_name+'.fits', spmask_final, overwrite=True)
    return spmask_final


def interpolate_spiral_arm(marker, r_bins, phi_bins, phi_min, phi_max):
    """
    Interpolate the spiral arm from marker data.
    
    Parameters:
    - marker: 2D array with markers
    - r_bins: radial bins
    - phi_bins: azimuthal angle bins
    
    Returns:
    - r_fine: interpolated radius values
    - phi_fine: interpolated azimuthal angle values
    """

    # Get marker indices
    marker_indices = np.argwhere(marker > 0)

    # Convert indices to (r, phi) using bin centers
    r_idx, phi_idx = marker_indices[:, 0], marker_indices[:, 1]
    r_vals = (r_bins[r_idx] + r_bins[r_idx+1]) / 2
    phi_vals = (phi_bins[phi_idx] + phi_bins[phi_idx+1]) / 2

    # Select only markers with phi between phi_min and phi_max
    mask = (phi_vals >= phi_min) & (phi_vals <= phi_max)
    r_sel = r_vals[mask]
    phi_sel = phi_vals[mask]

    # Sort by phi for smooth interpolation
    sort_idx = np.argsort(phi_sel)
    r_sorted = r_sel[sort_idx]
    phi_sorted = phi_sel[sort_idx]

    # Interpolate phi as a function of r
    if len(r_sorted) > 3:  # Need at least 4 points for spline
        from scipy.interpolate import UnivariateSpline
        spline = UnivariateSpline(phi_sorted, r_sorted, s=2)
        phi_fine = phi_sorted  # np.linspace(phi_sorted.min(), phi_sorted.max(), 100)
        r_fine = spline(phi_sorted)
        return spline
    else:
        print("Not enough marker points in the selected phi range for interpolation.")
        return None