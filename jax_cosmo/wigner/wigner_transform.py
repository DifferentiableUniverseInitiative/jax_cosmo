from scipy.interpolate import interp1d,interp2d,RectBivariateSpline
from scipy.optimize import fsolve
import numpy as np
import itertools
from scipy.special import eval_jacobi as jacobi
from scipy.special import binom,jn,loggamma
from multiprocessing import Pool,cpu_count
from functools import partial



def wigner_d(m1,m2,theta,l,l_use_bessel=1.e4):
    """
    Function to compute wigner matrices used in wigner transforms.
    """
    l0=np.copy(l)
    if l_use_bessel is not None:
    #FIXME: This is not great. Due to a issues with the scipy hypergeometric function,
    #jacobi can output nan for large ell, l>1.e4
    # As a temporary fix, for ell>1.e4, we are replacing the wigner function with the
    # bessel function. Fingers and toes crossed!!!
    # mpmath is slower and also has convergence issues at large ell.
    #https://github.com/scipy/scipy/issues/4446
    
        l=np.atleast_1d(l)
        x=l<l_use_bessel
        l=np.atleast_1d(l[x])
    k=np.amin([l-m1,l-m2,l+m1,l+m2],axis=0)
    a=np.absolute(m1-m2)
    lamb=0 #lambda
    if m2>m1:
        lamb=m2-m1
    b=2*l-2*k-a
    d_mat=(-1)**lamb
    d_mat*=np.sqrt(binom(2*l-k,k+a)) #this gives array of shape l with elements choose(2l[i]-k[i], k[i]+a)
    d_mat/=np.sqrt(binom(k+b,b))
    d_mat=np.atleast_1d(d_mat)
    x=k<0
    d_mat[x]=0

    d_mat=d_mat.reshape(1,len(d_mat))
    theta=theta.reshape(len(theta),1)
    d_mat=d_mat*((np.sin(theta/2.0)**a)*(np.cos(theta/2.0)**b))
    x=d_mat==0
    d_mat*=jacobi(k,a,b,np.cos(theta)) #l
    d_mat[x]=0
    
    if l_use_bessel is not None:
        l=np.atleast_1d(l0)
        x=l>=l_use_bessel
        l=np.atleast_1d(l[x])
#         d_mat[:,x]=jn(m1-m2,l[x]*theta)
        d_mat=np.append(d_mat,jn(m1-m2,l*theta),axis=1)
    return d_mat

def wigner_d_parallel(m1,m2,theta,l,ncpu=None,l_use_bessel=1.e4):
    """
    Compute wigner matrix in parallel.
    """
    if ncpu is None:
        ncpu=cpu_count()
    p=Pool(ncpu)
    d_mat=np.array(p.map(partial(wigner_d,m1,m2,theta,l_use_bessel=l_use_bessel),l))
    p.close()
    p.join()
    return d_mat[:,:,0].T


class wigner_transform():
    def __init__(self,theta=[],l=[],s1_s2=[(0,0)],logger=None,ncpu=None,**kwargs):
        self.name='Wigner'
        self.logger=logger
        self.l=l
        self.grad_l=np.gradient(l)
        self.norm=(2*l+1.)/(4.*np.pi) 
        self.wig_d={}
        # self.wig_3j={}
        self.s1_s2s=s1_s2
        self.theta={}
        # self.theta=theta
        for (m1,m2) in s1_s2:
            self.wig_d[(m1,m2)]=wigner_d_parallel(m1,m2,theta,self.l,ncpu=ncpu)
#             self.wig_d[(m1,m2)]=wigner_d_recur(m1,m2,theta,self.l)
            self.theta[(m1,m2)]=theta #FIXME: Ugly

    def reset_theta_l(self,theta=None,l=None):
        """
        In case theta ell values are changed. This can happen when we implement the binning scheme.
        """
        if theta is None:
            theta=self.theta
        if l is None:
            l=self.l
        self.__init__(theta=theta,l=l,s1_s2=self.s1_s2s,logger=self.logger)

    def cl_grid(self,l_cl=[],cl=[],taper=False,**kwargs):
        """
        Interpolate a given C_ell onto the grid of ells for which WT is intialized. 
        This is to generalize in case user doesnot want to compute C_ell at every ell.
        Also apply tapering if needed.
        """
        if taper:
            sself.taper_f=self.taper(l=l,**kwargs)
            cl=cl*taper_f
        # if l==[]:#In this case pass a function that takes k with kwargs and outputs cl
        #     cl2=cl(l=self.l,**kwargs)
        # else:
        cl_int=interp1d(l_cl,cl,bounds_error=False,fill_value=0,
                        kind='linear')
        cl2=cl_int(self.l)
        return cl2

    def cl_cov_grid(self,l_cl=[],cl_cov=[],taper=False,**kwargs):
        """
        Interpolate a given C_ell covariance onto the grid of ells for which WT is intialized. 
        This is to generalize in case user doesnot want to compute C_ell at every ell.
        Also apply tapering if needed.
        """
        if taper:#FIXME there is no check on change in taper_kwargs
            if self.taper_f2 is None or not np.all(np.isclose(self.taper_f['l'],cl)):
                self.taper_f=self.taper(l=l,**kwargs)
                taper_f2=np.outer(self.taper_f['taper_f'],self.taper_f['taper_f'])
                self.taper_f2={'l':l,'taper_f2':taper_f2}
            cl=cl*self.taper_f2['taper_f2']
        if l_cl_cl==[]:#In this case pass a function that takes k with kwargs and outputs cl
            cl2=cl_cov(l=self.l,**kwargs)
        else:
            cl_int=RectBivariateSpline(l_cl,l_cl,cl_cov,)#bounds_error=False,fill_value=0,
                            #kind='linear')
                    #interp2d is slow. Make sure l_cl is on regular grid.
            cl2=cl_int(self.l,self.l)
        return cl2

    def projected_correlation(self,l_cl=[],cl=[],s1_s2=[],taper=False,wig_d=None,**kwargs):
        """
        Get the projected correlation function from given c_ell.
        """
        if wig_d is None: #when using default wigner matrices, interpolate to ensure grids match.
            cl2=self.cl_grid(l_cl=l_cl,cl=cl,taper=taper,**kwargs)
            w=np.dot(self.wig_d[s1_s2]*self.grad_l*self.norm,cl2)
        else:
            w=np.dot(wig_d,cl)
        return self.theta[s1_s2],w

    def projected_covariance(self,l_cl=[],cl_cov=[],s1_s2=[],s1_s2_cross=None,
                            taper=False,**kwargs):
        if s1_s2_cross is None:
            s1_s2_cross=s1_s2
        #when cl_cov can be written as vector, eg. gaussian covariance
        cl2=self.cl_grid(l_cl=l_cl,cl=cl_cov,taper=taper,**kwargs)
        cov=np.einsum('rk,k,sk->rs',self.wig_d[s1_s2]*np.sqrt(self.norm),cl2*self.grad_l,
                    self.wig_d[s1_s2_cross]*np.sqrt(self.norm),optimize=True)
        #FIXME: Check normalization
        #FIXME: need to allow user to input wigner matrices.
        return self.theta[s1_s2],cov

    def projected_covariance2(self,l_cl=[],cl_cov=[],s1_s2=[],s1_s2_cross=None,
                                taper=False,**kwargs):
        #when cl_cov is a 2-d matrix
        if s1_s2_cross is None:
            s1_s2_cross=s1_s2
        cl_cov2=cl_cov  #self.cl_cov_grid(l_cl=l_cl,cl_cov=cl_cov,s1_s2=s1_s2,taper=taper,**kwargs)

        cov=np.einsum('rk,kk,sk->rs',self.wig_d[s1_s2]*np.sqrt(self.norm)*self.grad_l,cl_cov2,
                    self.wig_d[s1_s2_cross]*np.sqrt(self.norm),optimize=True)
#         cov=np.dot(self.wig_d[s1_s2]*self.grad_l*np.sqrt(self.norm),np.dot(self.wig_d[s1_s2_cross]*np.sqrt(self.norm),cl_cov2).T)
        # cov*=self.norm
        #FIXME: Check normalization
        return self.theta[s1_s2],cov

    def taper(self,l=[],large_k_lower=10,large_k_upper=100,low_k_lower=0,low_k_upper=1.e-5):
        #FIXME there is no check on change in taper_kwargs
        if self.taper_f is None or not np.all(np.isclose(self.taper_f['k'],k)):
            taper_f=np.zeros_like(k)
            x=k>large_k_lower
            taper_f[x]=np.cos((k[x]-large_k_lower)/(large_k_upper-large_k_lower)*np.pi/2.)
            x=k<large_k_lower and k>low_k_upper
            taper_f[x]=1
            x=k<low_k_upper
            taper_f[x]=np.cos((k[x]-low_k_upper)/(low_k_upper-low_k_lower)*np.pi/2.)
            self.taper_f={'taper_f':taper_f,'k':k}
        return self.taper_f

    def diagonal_err(self,cov=[]):
        return np.sqrt(np.diagonal(cov))

    def skewness(self,l_cl=[],cl1=[],cl2=[],cl3=[],s1_s2=[],taper=False,**kwargs):
        """
        Because we can do 6 point functions as well :). 
        """
        cl1=self.cl_grid(l_cl=l_cl,cl=cl1,s1_s2=s1_s2,taper=taper,**kwargs)
        cl2=self.cl_grid(l_cl=l_cl,cl=cl2,s1_s2=s1_s2,taper=taper,**kwargs)
        cl3=self.cl_grid(l_cl=l_cl,cl=cl3,s1_s2=s1_s2,taper=taper,**kwargs)
        skew=np.einsum('ji,ki,li',self.wig_d[s1_s2],self.wig_d[s1_s2],
                        self.wig_d[s1_s2]*cl1*cl2*cl3)
        skew*=self.norm
        #FIXME: Check normalization
        return self.theta[s1_s2],skew

if __name__ == "__main__":
    l0=np.arange(2000)+1
    cl=1./l0**2 #replace this with actual power spectra
    th_min=2.5/60
    th_max=250./60
    th=np.logspace(np.log10(th_min),np.log10(th_max),40)

    WT_kwargs={'l':l0,'theta':th,'s1_s2':[(2,2),(2,-2),(0,0)]}
    WT=wigner_transform(**WT_kwargs)

    th,xi_plus=WT.projected_correlation(l_cl=l0,cl=cl,s1_s2=(2,2))
    th,xi_minus=WT.projected_correlation(l_cl=l0,cl=cl,s1_s2=(2,-2))
    th,xi_g=WT.projected_correlation(l_cl=l0,cl=cl,s1_s2=(0,0))

    print(xi_plus,xi_minus,xi_g)