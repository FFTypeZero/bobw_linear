import numpy as np
import matplotlib.pyplot as plt


def fw_XY_combi(Y, reg_l2=0, iters=1000, step_size=.5, viz_step = 9999, initial=None, line_search=False ):
    n,d = Y.shape
    supp = np.sum(Y, axis=0) > 0
    #design = np.random.rand(d)
    #design /= design.sum()  
    design = np.zeros(d)
    design[supp] = 1/sum(supp)
    grad_norms = []
    history = []
    I = np.eye(d)
    for count in range(1, iters): 
        def f(design):
            l_inv = 1/(design+reg_l2)
            rho = Y@l_inv
            return rho, l_inv
        rho, l_inv = f(design)
        if count % (viz_step) == 0: 
            print('fw_XY_combi:',np.sqrt(np.max(rho)), np.min(l_inv))
        y_opt = Y[np.argmax(rho),:]
        g = y_opt * l_inv
        g = -g * g
        eta = step_size/(count+2)
        imin = np.argmin(g)
        old_design = design
        
        if line_search == False:
            design = (1-eta)*design+eta*I[imin]
        else:
            argeta = -1
            argf = float('Inf')
            for s in np.linspace(eta/2,eta*10, 20):
                d = (1-s)*design+s*I[imin]
                v = max(f(d)[0]) 
                if v < argf:
                    argeta = s
                    argf = v
            design = (1-eta)*design+eta*I[imin]
        
        grad_norms.append(-g.T@(I[imin]-old_design))
        history.append(np.sqrt(np.max(rho)))
        if count % (viz_step) == 0:
            fig, ax = plt.subplots(1,3)
            ax[0].plot(grad_norms)
            ax[1].plot(design)
            ax[2].plot(history)
            plt.show()
            print('fw_XY_combi:',np.sqrt(max(rho)))
    return design, max(rho)


def fw_XY(X, Y, reg_l2=0, iters=1000, step_size=.5, viz_step = 10000, initial=None):
    n = X.shape[0]
    d = X.shape[1]
    design = np.ones(n)
    design /= design.sum()  
    eta = step_size
    grad_norms = []
    history = []
    for count in range(1, iters): 
        A_inv = np.linalg.pinv(X.T@np.diag(design)@X + reg_l2*np.eye(d))        
        rho = np.diag(Y@A_inv@Y.T)
        y_opt = Y[np.argmax(rho),:]
        g = y_opt @ A_inv @ X.T
        g = -g * g
        eta = step_size/(count+2)
        imin = np.argmin(g)
        design = (1-eta)*design+eta*np.eye(n)[imin]
        grad_norms.append(np.linalg.norm(g - np.sum(g)/n*np.ones(n)))
        if count % (viz_step) == 0:
            history.append(np.sqrt(np.max(rho)))
            fig, ax = plt.subplots(1,3)
            ax[0].plot(grad_norms)
            ax[1].plot(design)
            ax[2].plot(history)
            plt.show()
    return design, max(rho)

