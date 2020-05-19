import numpy as np
import astropy.coordinates as coord
import astropy.units as u
from astropy.coordinates.matrix_utilities import matrix_product

def arcsec_to_pc(x, D):
	return (x.to(u.rad)).value*D.to(u.pc)

def pc_to_arcsec(x, D):
	return ((x/D)*u.rad).to(u.arcsec)

def rotate(inc):
	M = np.array(
			[[1, 0, 0],
			[0, -np.cos(inc), np.sin(inc)],
			[0, np.sin(inc), np.cos(inc)]
			])
	return M

def projected_to_aligned3D(x, y, z, inc):
	unit = x.unit
	x = x.to(unit)
	y = y.to(unit)
	z = z.to(unit)
	xyz = np.array([x.value,
					y.value,
					z.value])*unit
	M = rotate(inc).T
	XYZ = matrix_product(M, xyz)
	X = XYZ[0,:]
	Y = XYZ[1,:]
	Z = XYZ[2,:]
	return X,Y,Z

def aligned3D_to_projected(X, Y, Z, inc):
	XYZ = np.array([X.to('pc').value,
					Y.to('pc').value,
					Z.to('pc').value])*u.pc
	M = rotate(inc)
	xyz = matrix_product(M, XYZ)
	x = xyz[0,:]
	y = xyz[1,:]
	z = xyz[2,:]
	return x, y, z

def deproject_coords(x, y, z_pc, D, inc):
	x, y, z_pc = np.atleast_1d(x), np.atleast_1d(y), np.atleast_1d(z_pc)
	x_pc = arcsec_to_pc(x, D)
	y_pc = arcsec_to_pc(y, D)
	X, Y, Z = projected_to_aligned3D(x_pc, y_pc, z_pc, inc)
	return X, Y, Z

def project_coords(X, Y, Z, D, inc):
	X, Y, Z = np.atleast_1d(X), np.atleast_1d(Y), np.atleast_1d(Z)
	x_pc, y_pc, z_pc = aligned3D_to_projected(X, Y, Z, inc)
	x = pc_to_arcsec(x_pc, D)
	y = pc_to_arcsec(y_pc, D)
	return x, y, z_pc

def cylin_to_aligned3D_vels(vR, vphi, vz, phi):
	M = np.array([
			[np.cos(phi), -np.sin(phi), 0],
			[np.sin(phi), np.cos(phi), 0],
			[0, 0, 1]
			])
	vR, vphi, vz = np.atleast_1d(vR), np.atleast_1d(vphi), np.atleast_1d(vz)
	v_cylin = np.array([vR.to('km/s').value,
						vphi.to('km/s').value,
						vz.to('km/s').value])
	v_cartesian = matrix_product(M, v_cylin)
	return v_cartesian * u.km/u.s

def cylin_to_aligned3D_vels_phiarray(vR, vphi, vz, phi):
	vR, vphi, vz = np.atleast_1d(vR), np.atleast_1d(vphi), np.atleast_1d(vz)
	v_cylin = np.array([vR.to('km/s').value,
						vphi.to('km/s').value,
						vz.to('km/s').value])
	N = len(phi)
	M = np.array([
			[np.cos(phi).value, -np.sin(phi).value, np.zeros(N)],
			[np.sin(phi).value, np.cos(phi).value, np.zeros(N)],
			[np.zeros(N), np.zeros(N), np.ones(N)]
			])
	v_cartesian = np.einsum('ij...,j...', M, v_cylin)
	return v_cartesian.T * u.km/u.s

def aligned3D_to_cylin(X, Y, Z, vX, vY, vZ):
	R = np.sqrt(X**2 + Y**2)
	phi = np.arctan2(Y, X)
	N = len(phi)
	M = np.array([
			[np.cos(phi).value, np.sin(phi).value, np.zeros(N)],
			[-np.sin(phi).value, np.cos(phi).value, np.zeros(N)],
			[np.zeros(N), np.zeros(N), np.ones(N)]
			])
	vX, vY, vZ = np.atleast_1d(vX), np.atleast_1d(vY), np.atleast_1d(vZ)
	v_cartesian = np.array([vX.to('km/s').value,
							vY.to('km/s').value,
							vZ.to('km/s').value])
	v_cylin = np.einsum('ij...,j...', M, v_cartesian)
	vR, vphi, vZ = v_cylin.T * u.km/u.s
	return R, phi, Z, vR, vphi, vZ

def cylin_to_aligned3D(R, phi, Z, vR, vphi, vZ):
	tanphi = np.tan(phi)
	X = R / np.sqrt(1. + tanphi**2.)
	Y = X * tanphi
	v_cartesian = cylin_to_aligned3D_vels_phiarray(vR, vphi, vZ, phi)
	vX, vY, vZ = v_cartesian
	return X, Y, Z, vX, vY, vZ


#
