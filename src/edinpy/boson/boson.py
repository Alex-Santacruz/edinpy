_dofs = None # global list of constructed degrees of freedom (DoF)
_N_vector = None # global list of attributes <dof>.N for nD -> 1D mapping
_fockspace = None # global var for constructed Fock space

def clear():
    ''' Clear all the degrees of freedom and fock space stored as global variables. It is
        necessary to call this function when you want to change the model.
    '''
    global _dofs, _N_vector, _fockspace
    _dofs = None
    _N_vector = None
    _fockspace = None

_normal_chars = "0123456789,"
_subscript_chars = "₀₁₂₃₄₅₆₇₈₉,"
#_superscript_chars = "⁰¹²³⁴⁵⁶⁷⁸⁹"
_mapping = str.maketrans(_normal_chars, _subscript_chars)

def _subscripts(text):
    ''' Converts subscript from normal text numbers to subscript numbers.
    '''
    converted_text = text.translate(_mapping)
    return converted_text

class DoF:
    ''' 
    Create a degree of freedom.
        
    Parameters
    ----------
    N_eigenvalues : int
        Number of eigenvalues of the associated observable, e.g. electron
        spin has 2 eigenvalues, a chain of 5 atoms has 5 position eigenvalues.
    name : string
        Name of the observable.
    '''

    def __init__(self, N_eigenvalues, name = None):
        self._N = N_eigenvalues # number of eigvals of observable
        self._name = name # given name of observable
        self._add_dof()

    def _add_dof(self):
        ''' 
        Add DoF to global vars lists.
        '''
        global _dofs, _N_vector

        # add attr N to a global list used to calculate effective sites
        if _N_vector is not None:
            _N_vector.append(_N_vector[-1] * _dofs[-1].N)
        else:
            _N_vector = [1]

        # add DoF to a global list
        if _dofs is not None:
            _dofs.append(self)
        else:
            _dofs = [self]

        self._label = len(_dofs) - 1 # numerical label of DoF

    @property
    def N(self):
        '''
        Number of eigenvalues set by parameter `N_eigenvalues`.
        :type: int
        '''
        return self._N
    
    @N.setter
    def N(self, N):
        self._N = N
    
    @property
    def name(self):
        ''' 
        Name given to the degree of freedom
        :type: string
        '''
        return self._name
    
    @name.setter
    def name(self, name):
        self._name = name
    
    @property
    def label(self):
        ''' 
        Numerical label given to the degree of freedom
        :type: int
        '''
        return self._label
    
    @label.setter
    def label(self, label):
        self._label = label

class model:
    ''' 
    Construct a model with the stored degrees of freedom.

    Parameters
    ----------
    Nb : int
        Number of bosons.
    auto : bool
        If auto is True then the Fock space is calculated automatically, otherwise
        the user must use the build() method to calculate it manually.           
    '''

    def __init__(self, Nb, auto = True):
        global _dofs, _N_vector, _Nb

        self._dofs = _dofs
        self.Nvec = _N_vector
        self.Nb = _Nb = Nb

        if auto is True:
            self.build()
        
    def build(self):
        '''
        Build the Fock space with the given number of fermions and degrees of freedom.
        '''
        from numpy import prod
        from math import factorial as fact
        global _Neff, _Nbasis, _fockspace
        
        self._Neff = _Neff = prod([dof.N for dof in self._dofs])
        self._Nbasis = _Nbasis = int( fact(self.Nb + self._Neff - 1) / (fact(self.Nb) * fact(self._Neff - 1)) )
        self._fockspace = _fockspace = bosonspace()
        _fockspace.build()
        self._fockspace.build()

    def activate(self):
        ''' 
        Set this model as the active instance.  
        '''
        global _dofs, _N_vector
        global _Neff, _Nb, _Nbasis, _fockspace

        _dofs = self._dofs
        _N_vector = self.Nvec
        _Neff = self._Neff
        _Nb = self.Nb
        _Nbasis = self._Nbasis
        _fockspace = self._fockspace

    @property
    def Neff(self):
        '''
        Number of effective sites.
        :type: int
        '''
        return self._Neff
    
    @Neff.setter
    def Neff(self, Neff):
        self._Neff = Neff

    @property
    def Nbasis(self):
        '''
        Number of basis states in the Fock space.
        :type: int
        '''
        return self._Nbasis
    
    @Nbasis.setter
    def Nbasis(self, Nbasis):
        self._Nbasis = Nbasis

    @property
    def fockspace(self):
        '''
        Class instance for the relevant Fock space of the current model.
        :type: fermionspace
        '''
        return self._fockspace
    
    @fockspace.setter
    def fockspace(self, fockspace):
        self._fockspace = fockspace

    @property
    def dofs(self):
        '''
        List of instances of degrees of freedom.
        :type: list
        '''
        return self._dofs
    
    @dofs.setter
    def dofs(self, dofs):
        self._dofs = dofs

class fockstate():
    ''' 
    Create a bosonic Fock state.

    Parameters
    ----------
    state : np.array
        Occupation numbers of the effective sites represented as a numpy array.
    amp : scalar
        Probability amplitude of the state. Default value is 1.
    index : int
        Index of the state. Used when constructing a Fock space. Default is None.
    '''

    def __init__(self, state, amp = 1, index = None):
        from numpy import csingle
        self.state = state
        self.amp = csingle(amp)
        self.index = index
        self._tag = self.calc_tag()
        self.bra = False

    def __pos__(self):
        return self

    def __neg__(self):
        return fockstate(self.state, amp = -self.amp, index = self.index)
    
    def __add__(self,bs):

        if isinstance(bs, fockstate):
            return statesum([self,bs])
        
        if isinstance(bs, null):
            return self
        
        if isinstance(bs, statesum):
            return statesum([self] + bs.states)
        
        raise NotImplementedError
    
    def __sub__(self,bs):

        if isinstance(bs, fockstate):
            return statesum([self,-bs])
        
        if isinstance(bs, null):
            return self
        
        if isinstance(bs, statesum):
            negstates = [-s for s in bs.states]
            return statesum([self] + negstates)
        
        raise NotImplementedError
    
    def __mul__(self,obj):
        
        if isinstance(obj, fockstate):
            if obj.tag == self._tag:
                if self.bra is True:
                    return self.amp * obj.amp
                else:
                    return self.amp.conj() * obj.amp
            else:
                return 0
            
        if isinstance(obj, null):
            return 0
        
        if isinstance(obj, scalar):
            if self.bra is False:
                amp = obj.value * self.amp.conj()
            else:
                amp = obj.value * self.amp
            newfs = fockstate( self.state, amp = amp)
            newfs.bra = True
            return newfs
        
        if isinstance(obj, operator):
            product =  obj.dag*self
            product.bra = True
            if self.bra is False:
                product.amp = product.amp.conj()
            return product
        
        if isinstance(obj, operatorsum):
            sum = self*obj.os[0]
            for i in range(1,len(obj.os)):
                sum += self*obj.os[i]
            return sum
        
        if isinstance(obj, operatorproduct):
            product = self*obj.op[0]
            for i in range(1,len(obj.op)):
                product *= obj.op[i]
            return product
        
        if isinstance(obj,statesum):
            sum = 0
            for s in obj.states:
                sum += self*s
            return sum
        
        raise NotImplementedError

    def __eq__(self,bs):
        if bs.tag == self._tag:
            return True
        else:
            return False
        
    def __str__(self):
        return self.string
    
    def __format__(self, spec = None):
        return self.string
    
    def is_occupied(self, _calc_effsite):
        '''
        Determine whether the effective site is occupied.
        :param _calc_effsite: effective site index
        :type _calc_effsite: int
        :rtype: bool
        '''
        return self.state[_calc_effsite] != 0

    def calc_tag(self):
        """ 
        Calculate the tag of the fock state.
        :rtype: np.float
        """
        from numpy import sqrt
        tag = 0
        for i in range(_Neff):
            tag += sqrt(3 + 100 * (i+1) ) * self.state[i]
        return tag
    
    @property
    def string(self):
        ''' 
        Human readable representation of Fock state.
        :rtype: str
        '''
        string = ''
        for i in range(len(self.state)-1):
            string += f'{self.state[i]},'
        string += f'{self.state[-1]}'

        return f'{self.amp}|{string}>'
    
    @property
    def tag(self):
        '''
        A numerical tag used to uniquely identify the Fock state.
        :type: float
        '''
        return self._tag
    
    @tag.setter
    def tag(self, tag):
        self._tag = tag
    
class vacuum(fockstate):
    ''' Create a vacuum state.

        Parameters
        ----------
        amp : scalar
            Probability amplitude of the state. Default value is `1`.
    '''

    def __init__(self, amp = 1):
        from numpy import csingle
        self.amp = csingle(amp)
        self.state = 0
        self.index = -1
        self.bra = False

    def __mul__(self, obj):
        
        if isinstance(obj, fockstate):

            if isinstance(obj, vacuum):
                if self.bra is True:
                    return self.amp * obj.amp
                else:
                    return self.amp.conj() * obj.amp
            
            return 0
        
        if isinstance(obj, null):
            return 0
        
        if isinstance(obj, scalar):
            if self.bra is False:
                self.amp = obj.value * self.amp.conj()
            else:
                self.amp = obj.value * self.amp
            self.bra = True
            return self
        
        if isinstance(obj, operator):
            product =  obj.dag * self
            product.bra = True
            if self.bra is False:
                product.amp = product.amp.conj()
            return product
        
        if isinstance(obj, operatorsum):
            sum = self*obj.os[0]
            for i in range(1,len(obj.os)):
                sum += self*obj.os[i]
            return sum
        
        if isinstance(obj, operatorproduct):
            product = self*obj.op[0]
            for i in range(1,len(obj.op)):
                product = product * obj.op[i]
            return product
        
        if isinstance(obj,statesum):
            sum = 0
            for s in obj.states:
                sum += self*s
            return sum
        
        raise NotImplementedError

    def __eq__(self,ket):
        if isinstance(ket,vacuum):
            return True
        else:
            return False
    
    @property
    def string(self):
        ''' 
        Human readable representation of Fock state.
        :rtype: str
        '''
        return f'{self.amp}|{0}>'
    
class null():
    ''' 
    Additive identity for Fock states and operators.
    '''

    def __init__(self):
        from numpy import csingle
        self.amp = csingle(0)
        self.state = 0
        self.index = None

    def __add__(self,bs):
        return bs

    def __mul__(self,obj):

        if isinstance(obj, fockstate):
            return 0
        
        if isinstance(obj, scalar):
            return self
        
        if isinstance(obj, (operator, operatorsum, operatorproduct)):
            return self
        
        if isinstance(obj, statesum):
            return 0
        
        raise NotImplementedError
    
    def __eq__(self,bs):
        if isinstance(bs,null):
            return True
        else:
            return False
        
    def __str__(self):
        return self.string
    
    def __format__(self, spec = None):
        return self.string
    
    @property
    def string(self):
        ''' 
        Human readable representation of Fock state.
        :rtype: str
        '''
        return '0'
    
class statesum:
    ''' 
    Sum of Fock states.

    Parameters
    ----------
    stateslist : list
        List containing Fock states.
    '''

    def __init__(self, stateslist):
        self.states = stateslist

    def __add__(self, obj):

        if isinstance(obj, fockstate):
            states = [s for s in self.states]
            states.append(obj)
            return statesum(states)
        
        if isinstance(obj, operator):
            sum = self._os[0]*obj
            for i in range(1,len(self._os)):
                sum += self._os[i]*obj
            return sum
        
        if isinstance(obj, operatorsum):
            sum = self*obj.os[0]
            for i in range(1,len(obj.os)):
                sum += self*obj.os[i]
            return sum
        
        if isinstance(obj, operatorproduct):
            product = self*obj.op[0]
            for i in range(1,len(obj.op)):
                product *= obj.op[i]
            return product
        
        if isinstance(obj, statesum):
            return statesum(self.states + obj.states)
        
        if isinstance(obj, null):
            return self
        
        raise NotImplementedError
        
    def __mul__(self,obj):

        if isinstance(obj, fockstate):
            sum = 0
            for bra in self.states:
                sum += bra*obj
            return sum
        
        if isinstance(obj, scalar):
            sum = self._os[0] * obj
            for i in range(1, len(self._os[0])):
                sum += self._os[i] * obj
            return sum
        
        if isinstance(obj, statesum):
            sum = 0
            for bra in self.states:
                for ket in obj.states:
                    sum += bra*ket
            return sum
        
        raise NotImplementedError
        
    def __str__(self):
        return self.string

    def __format__(self,spec):
        return self.string
    
    @property
    def string(self):
        string = ''
        for i in range(len(self.states)-1):
            string += str(self.states[i]) + ' + '
        string += str(self.states[-1])
        return string

class bosonspace:
    ''' 
    Create a bosonic Fock space.
    '''

    def __init__(self):
        global _Nbasis, _Neff, _Nb

        self._Nbasis = _Nbasis
        self._Neff = _Neff
        self.Nb = _Nb
        self._tags = None

    def __contains__(self, s):
        from bisect import bisect_left
        
        i = bisect_left(self._tags, s.tag)
        if i != len(self._tags) and self._tags[i] == s.tag:
            return True, i
        else:
            return False

    contains = __contains__

    def build(self):
        '''
        Build the fock space basis vectors.
        '''
        self.basis  = [0 for i in range(self._Nbasis)]
        self._tags  = [0 for i in range(self._Nbasis)]
        self.basis[0] = fockstate( self.firstvector(), index = 0) # calculate the first state
        self._tags[0] = self.basis[0].tag
        # calculate the remaining states in lexicographical order
        for i in range(1, self._Nbasis):
            self.basis[i] = fockstate( self.nextvector(self.basis[i-1].state), index = i)
            self._tags[i] = self.basis[i].tag
     
    def firstvector(self):
        """ 
        Generate the first bosonic occupation vector in lexicographic ordering corresponding to the ket bv = |Nb,0,0,...,0> with dim(bv) = Neff.
        """
        from numpy import zeros
        bv = zeros(self._Neff, dtype = int) # fill the first effective site with Nb bosons
        bv[0] = self.Nb
        return bv
    
    def findk(self, bv):
        """ 
        Programmatically find the index k which generates the next basis vector in lexicographical order.
        Algorithm statement: Find k such that n_k =/= 0 and n_i = 0 for k+1<=i<=M-1. If there is no such k 
        then n is the last lexicographic basis vector. 
        
        Source - J M Zhang and R X Dong 2010 Eur. J. Phys. 31 591

        :param bv: a bosonic occupation vector
        :type bv: np.array
        """
        # Find k such that n_k =/= 0 and n_i = 0 for k+1 <= i <= M-1
        for k in range(self._Neff-1): # For every site except the last site,
            found_k = True
            if bv[k] != 0: # find the first non-zero element (NZE) from left to right.
                for i in range(k+1,self._Neff-1): # For every element to the right of NZE except the last site,
                    if bv[i] != 0: # if there are NZE to the right
                        found_k = False
                        break # then break loop and check next index k.
                # If loop not broken then n_i =0 for i=k+1,...,M-1, then k exists.
                if found_k:
                    return k
        # If there is no such k then n is the last lexicographic basis vector.
        return
    
    def nextvector(self, bv):
        """ 
        Find the next bosonic occupation vector in lexicographical order.
        
        Source - J M Zhang and R X Dong 2010 Eur. J. Phys. 31 591

        :param bv: a bosonic occupation vector
        :type bv: np.array
        """
        from numpy import copy

        k = self.findk(bv) # k-index that generates next vector

        if k is None:
            return

        newbs = copy(bv) # next basis vector
        
        # subtract one from site k
        newbs[k] = bv[k] - 1
        
        # fill site k+1 with remaining number of particles
        sum = 0
        for i in range(k+1):
            sum += bv[i]
        newbs[k+1] = self.Nb + 1 - sum 
        
        # empty sites from k+2 to M
        for i in range(k+2, self._Neff):
            newbs[i] = 0
        
        # copy site filling for sites left of k
        for i in range(k - 1):
            newbs[i] = bv[i]

        return newbs
    
    def show(self):
        '''
        Print all the basis states.
        '''
        for s in self.basis:
            print(f'state {s.index}: {s}')

class scalar:
    ''' 
    Create a scalar.

    Parameters
    ----------
    value : scalar
        The numerical value of the scalar
    label : str
        A string label for the scalar
    '''
    
    def __init__(self, value, label):
        self.value = value
        self.label = label

    def __pos__(self):
        return self

    def __neg__(self):
            return scalar(-1, f'({-1})') * self

    def __add__(self, obj):

        if isinstance(obj, scalar):
            return operatorsum([self, obj])
        
        if isinstance(obj, operatorsum):
            os = [self]
            os += obj.os
            return operatorsum(os)

        if isinstance(obj, operatorproduct):
            return operatorsum([self, obj])
        
        if isinstance(obj, (int, float, complex)):
            sclr = scalar(obj, f'{obj}')
            return operatorsum([self, sclr])

        raise NotImplementedError
    
    def __radd__(self, obj):
        
        if isinstance(obj, (int, float, complex)):
            sclr = scalar(obj, f'{obj}')
            return operatorsum([sclr, self])
        
        raise NotImplementedError
    
    def __sub__(self, obj):
        negone = scalar(-1,f'({-1})')
        
        if isinstance(obj, scalar):
            negobj = negone * obj
            return operatorsum([self, negobj])
        
        if isinstance(obj, (int, float, complex)):
            
            sclr = scalar(obj, f'{obj}')
            sclr = negone * sclr
            return operatorsum([self, sclr])

        raise NotImplementedError
    
    def __rsub__(self, obj):
        negone = scalar(-1,f'({-1})')
        
        if isinstance(obj, (int, float, complex)):
            
            sclr = scalar(obj, f'{obj}')
            negself = negone * self
            return operatorsum([sclr, negself])

        raise NotImplementedError

    def __mul__(self, obj):
        
        if isinstance(obj, fockstate):
            
            if isinstance(obj, vacuum):
                obj.amp *= self.value
                return obj
            else:
                amp = self.value * obj.amp
                return fockstate(obj.state, amp = amp)
            
        if isinstance(obj, null):
            return obj
        
        if isinstance(obj, operator):
            prod = [self, obj]
            return operatorproduct(prod)

        if isinstance(obj, operatorproduct):
            prod = [self] + obj.op
            return operatorproduct(prod)  

        if isinstance(obj, operatorsum):
            sum = self * obj.os[0]
            for i in range(1,len(obj.os)):
                sum += self * obj.os[i]
            return sum
        
        if isinstance(obj, statesum):
            sum = self * obj.states[0]
            for i in range(1,len(obj.states)):
                sum += self * obj.states[i]
            return sum
        
        if isinstance(obj, scalar):
            value = self.value * obj.value
            label = self.label + obj.label
            return scalar(value, label)
        
        if isinstance(obj, (int, float, complex)):
            sclr = scalar(obj, f'{obj}')

            return operatorproduct([sclr, self])
        
        raise NotImplementedError
    
    def __rmul__(self, obj):
        
        if isinstance(obj, (int, float, complex)):
            sclr = scalar(obj, f'{obj}')
            return operatorproduct([sclr, self])
        
        raise NotImplementedError
    
    def __str__(self):
        return f'{self.value}'
    
    def __format__(self, spec):
        return f'{self.label}'
    
    @property
    def string(self):
        '''
        Return the label of the scalar.
        '''
        return f'{self.label}'
    
class operator:
    ''' 
    Create an annihilation operator.

    Parameters
    ----------
    indices : iterable or int
        The indices which the operator acts on. If `indices` is an `int` then it is assumed 
        to refer to the effective site.
    '''

    def __init__(self, indices):
        self._indices = indices
        self._calc_effsite()

    def __pos__(self):
        return self

    def __neg__(self, obj):
        negone = scalar(-1, f'(-1)')
        return negone * self
    
    def __add__(self, obj):

        if isinstance(obj, (operator, operatorproduct)):
            return operatorsum([self, obj])
        
        if isinstance(obj, operatorsum):
            os = [0 for i in range(len(obj.os)+1)]
            os[0] = self
            for i, o in enumerate(obj.os):
                os[i+1] = o
            return operatorsum(os)
        
        if isinstance(obj, scalar):
            return operatorsum([self, obj])
        
        if isinstance(obj, (int, float, complex)):
            sclr = scalar(obj, f'{obj}')
            return operatorsum([self, sclr])
        
        raise NotImplementedError
    
    def __radd__(self, obj):

        if isinstance(obj, (int, float, complex)):
            sclr = scalar(obj, f'{obj}')
            return operatorsum([sclr, self])
        
        raise NotImplementedError
    
    def __sub__(self, obj):
        
        negone = scalar(-1,f'({-1})')
        
        if isinstance(obj, (operator, operatorproduct)):
            return operatorsum([self, negone * obj])
        
        if isinstance(obj, operatorsum):
            os = [0 for i in range(len(obj.os)+1)]
            os[0] = self
            for i, o in enumerate(obj.os):
                os[i+1] = negone * o
            return operatorsum(os)
        
        if isinstance(obj, scalar):
            return operatorsum([self, negone * obj])
        
        if isinstance(obj, (int, float, complex)):
            sclr = scalar(obj, f'{obj}')
            return operatorsum([self, negone * sclr])
        
        raise NotImplementedError
    
    def __rsub__(self, obj):
        negone = scalar(-1,f'({-1})')
        if isinstance(obj, (int, float, complex)):
            sclr = scalar(obj, f'{obj}')
            return operatorsum([obj, negone * self])
        
        raise NotImplementedError

    def __mul__(self,obj):
        
        if isinstance(obj,fockstate):
            return self._action(obj)
        
        if isinstance(obj,null):
            return obj
        
        if isinstance(obj, scalar):
            return operatorproduct([obj, self])
        
        if isinstance(obj,operator):
            return operatorproduct([self,obj])

        if isinstance(obj,operatorsum):
            os = [0 for i in range(len(obj.os))]
            for i, o in enumerate(obj.os):
                os[i] = self*o
            return operatorsum(os)

        if isinstance(obj,operatorproduct):
            op = [ o for o in obj.op ]
            op = [self] + op
            return operatorproduct(op)

        if isinstance(obj,statesum):
            sum = self*obj.states[0]
            for i in range(1,len(obj.states)):
                sum += self*obj.states[i]
            return sum
        
        if isinstance(obj, (int, float, complex)):
            sclr = scalar(obj, f'{obj}')
            return operatorproduct([sclr,self])
        
        raise NotImplementedError
    
    def __rmul__(self, obj):
        
        if isinstance(obj, (int, float, complex)):
            sclr = scalar(obj, f'{obj}')
            return operatorproduct([sclr,self])
        
        raise NotImplementedError

    def __eq__(self, obj):
        raise NotImplementedError
        
    def __str__(self):
        return self.string
    
    def __format__(self,spec):   
        return self.string

    def _annihilate(self,bs):
        """ 
        Annihilate a particle in a Fock state.

        :param bs: a bosonic Fock state
        :type bs: fockstate
        """
        from numpy import sqrt, copy
        if isinstance(bs,vacuum):
            return null()
        
        # Annihilate fermion in i-th site
        if bs.is_occupied(self._site): # if the site occupation number n_i != 0
            
            newamp = bs.amp * sqrt(bs.state[self._site])
            newstate = copy(bs.state)
            newstate[self._site] -= 1
            if newstate.nonzero()[0].size == 0:
                return vacuum(amp = newamp)
            else:
                return fockstate(newstate, amp = newamp)
        else:
            # otherwise annihilate the state
            return null()
        
    def _action(self,bs):
        '''
        Action of the annihilation operator on a Fock state.

        :param bs: a bosonic Fock state
        :type bs: fockstate
        '''
        return self._annihilate(bs)
        
    def _checkdims(self):
        '''
        Check the dimensions of the indices.
        '''
        if len(self._indices) != len(_dofs):
            raise Exception(f'The number of indices {len(self._indices)} exceeds the number of degrees of freedom {len(_dofs)}.')
        
        for i,d in enumerate(_dofs):
            if self._indices[i] >= d.N:
                nameorlabel = d.name if (d.name != None) else d.label
                raise Exception(f'The index {self._indices[i]} at position {i} is out of bounds for DoF [{nameorlabel}] of size {d.N-1} (0 < index < {d.N}).')
            
    def _calc_effsite(self):
        '''
        Calculate the effective site corresponding the given indices.
        '''
        if isinstance(self._indices,int):
            if self._indices >= _Neff:
                raise Exception(f'The index {self._indices} is out of bounds for the number of effective sites {_Neff} (0 < index < {_Neff})')
            self._site = self._indices
            return
        
        from numpy import array
        self._checkdims()
        self._site = int(array(_N_vector).dot(self._indices))
        
    @property
    def dag(self):
        '''
        Return the Hermitian adjoint of the operator.
        '''
        return dagger(self._indices, self._site)
    
    @property
    def indices(self):
        '''
        The indices which the operator acts on.
        :rtype: iterable or int
        '''
        return self._indices
    
    @indices.setter
    def indices(self, indices):
        self._indices = indices
    
    @property
    def site(self):
        '''
        Effective site that the operator acts on.
        :rtype: int
        '''
        return self._site
    
    @site.setter
    def site(self, effsite):
        self._site = effsite
    
    @property
    def string(self):
        '''
        Return the string representing the operator.
        '''
        if isinstance(self._indices, int):
            string = str(self._indices)
        else:
            string = ''
            for i in range(len(self._indices)-1):
                string += str(self._indices[i]) + ','
            string += str(self._indices[-1])    
        return 'b' + _subscripts(string)
    
class dagger(operator):
    ''' 
    Hermitian adjoint of an operator.

    Parameters
    ----------
    indices : iterable or int
        The indices which the operator acts on. If `indices` is an `int` then it is assumed 
        to refer to the effective site.
    site : int
        Effective site that the operator acts on.
    '''

    def __init__(self, indices, site):
        self._indices = indices
        self._site = site
    
    def __format__(self,spec):
        return self.string

    def __eq__(self, obj):
        if isinstance(obj,dagger):
            if obj.site == self._site:
                return True
        return False
    
    def _create(self,bs):
        """ Creation operator acting on the i-th site. Notation: state s -> (s,p) where p is probability amplitude
        """
        from numpy import sqrt, copy

        if isinstance(bs,vacuum):
            from numpy import zeros
            global _Neff
            newstate = zeros(_Neff)
            newstate[self._site] += 1
            return fockstate( newstate, amp = bs.amp)
        
        newamp = bs.amp * sqrt(bs.state[self._site] + 1)
        newstate = copy(bs.state)
        newstate[self._site] += 1

        return fockstate(newstate, amp = newamp)
        
    def _action(self,obj):
        return self._create(obj)
        
    @property
    def dag(self):
        '''
        Return the Hermitian adjoint of the operator.
        '''
        return operator(self._indices)
    
    @property
    def string(self):
        '''
        Return the string representing the operator.
        '''
        if isinstance(self._indices, int):
            string = str(self._indices)
        else:
            string = ''
            for i in range(len(self._indices)-1):
                string += str(self._indices[i]) + ','
            string += str(self._indices[-1])    
        return 'b†' + _subscripts(string)
    
class operatorsum:
    ''' 
    Sum of operators.

    Parameters
    ----------
    operatorlist : list
        A list containing the operators that are summed.
    '''
    
    def __init__(self, operatorlist):
        self._os = operatorlist

    def __pos__(self):
        return self

    def __neg__(self):
        os = [-o for o in self._os]
        return operatorsum(os)

    def __add__(self, obj):

        if isinstance(obj, (scalar, operator, operatorproduct)):
            os = [o for o in self._os]
            os.append(obj)
            return operatorsum(os)
        
        if isinstance(obj, operatorsum):
            return operatorsum(self._os + obj.os)
        
        if isinstance(obj, (int, float, complex)):
            sclr = scalar(obj, f'{obj}')
            return self + sclr
        
        raise NotImplementedError
    
    def __radd__(self, obj):
        
        if isinstance(obj, (int, float, complex)):
            sclr = scalar(obj, f'{obj}')
            return sclr + self
        
        raise NotImplementedError
    
    def __sub__(self, obj):
        negone = scalar(-1,f'({-1})')

        if isinstance(obj, (scalar, operator, operatorproduct)):
            os = [o for o in self._os]
            os.append(negone * obj)
            return operatorsum(os)
        
        if isinstance(obj, operatorsum):
            negobj = negone * obj
            return operatorsum(self._os + negobj.os)
        
        if isinstance(obj, (int, float, complex)):
            sclr = negone * scalar(obj, f'{obj}')
            return self + sclr
        
        raise NotImplementedError
    
    def __rsub__(self, obj):
        negone = scalar(-1,f'({-1})')
        
        if isinstance(obj, (int, float, complex)):
            sclr = scalar(obj, f'{obj}')
            return sclr + negone*self
        
        raise NotImplementedError
        
    def __mul__(self, obj):
        
        if isinstance(obj, fockstate):
            sum = self._os[0]*obj
            for i in range(1,len(self._os)):
                sum += self._os[i]*obj
            return sum
        
        if isinstance(obj, scalar):
            sum = self._os[0] * obj
            for i in range(1, len(self._os)):
                sum += self._os[i] * obj
            return sum
        
        if isinstance(obj, (operator,operatorproduct)):
            os = [0 for i in range(len(self._os))]
            for i, o in enumerate(self._os):
                os[i] = o*obj
            return operatorsum(os)
        
        if isinstance(obj, operatorsum):
            sum = self._os[0]*obj
            for i in range(1,len(self._os)):
                sum += self._os[i]*obj
            return sum
        
        if isinstance(obj, statesum):
            sum = self._os[0]*obj
            for i in range(1,len(self._os)):
                sum += self._os[i]*obj
            return sum
        
        if isinstance(obj, (int, float, complex)):
            sclr = scalar(obj, f'{obj}')
            return self * sclr

        raise NotImplementedError
    
    def __rmul__(self, obj):
        
        if isinstance(obj, (int, float, complex)):
            sclr = scalar(obj, f'{obj}')
            return sclr * self

        raise NotImplementedError
        
    def __eq__(self,obj):
        raise NotImplementedError
        
    def __str__(self):
        return self.string
    
    def __format__(self, spec):
        return self.string
    
    @property
    def dag(self):
        '''
        Return the Hermitian adjoint of the sum operators.
        '''
        listdag = [0 for o in self._os]
        for i, o in enumerate(self._os):
            listdag[i] = o.dag
        return operatorsum(listdag)
    
    @property
    def os(self):
        '''
        Return the list of summed operators
        '''
        return self._os
    
    @os.setter
    def os(self, operatorlist):
        self._os = operatorlist
    
    @property
    def string(self):
        '''
        Return the string representing the sum of operators.
        '''
        string = ''
        for i in range(len(self._os)-1):
            string += self._os[i].string + ' + '
        string += self._os[-1].string
        return string

        
class operatorproduct:
    ''' 
    Product of operators.

    Parameters
    ----------
    operatorlist : list
        A list containing the operators that are summed.
    '''
            
    def __init__(self, operatorlist):
        self._op = operatorlist

    def __pos__(self):
        return self

    def __neg__(self):
        negone = scalar(-1, '(-1)')
        return operatorproduct([negone] + self._op)

    def __add__(self, obj):

        if isinstance(obj, (scalar, operator, operatorproduct)):
            return operatorsum([self, obj])
        
        if isinstance(obj, operatorsum):
            os = [0 for i in range(len(obj.os)+1)]
            os[0] = self
            for i, o in enumerate(obj.os):
                os[i+1] = o
            return operatorsum(os)
        
        if isinstance(obj, (int, float, complex)):
            sclr = scalar(obj, f'{obj}')
            return operatorsum([self, sclr])
        
        raise NotImplementedError
    
    def __radd__(self, obj):
        if isinstance(obj, (int, float, complex)):
            sclr = scalar(obj, f'{obj}')
            return operatorsum([sclr, self])
        
        raise NotImplementedError
    
    def __sub__(self, obj):
        negone = scalar(-1,f'({-1})')

        if isinstance(obj, (scalar, operator, operatorproduct)):
            return operatorsum([self, negone * obj])
        
        if isinstance(obj, operatorsum):
            negobj = negone * obj
            os = [self]
            os += negobj.os
            return operatorsum(os)
        
        if isinstance(obj, (int, float, complex)):
            sclr = negone * scalar(obj, f'{obj}')
            return operatorsum([self, sclr])
        
        raise NotImplementedError
    
    def __rsub__(self, obj):
        negone = scalar(-1,f'({-1})')
        if isinstance(obj, (int, float, complex)):
            sclr = scalar(obj, f'{obj}')
            negself = negone * self
            return operatorsum([sclr, negself])
        
        raise NotImplementedError

    def __mul__(self,obj):

        if isinstance(obj, fockstate):
            product = self._op[-1]*obj
            for i in range(len(self._op)-2,-1,-1):
                product = self._op[i]*product
            return product
        
        if isinstance(obj, scalar):
            return operatorproduct([obj] + self._op)

        if isinstance(obj, operator):
            op = [o for o in self._op]
            op.append(obj)
            return operatorproduct(op)

        if isinstance(obj, operatorproduct):
            return operatorproduct(self._op + obj.op)

        if isinstance(obj, operatorsum):
            os = [0 for i in range(len(obj.os))]
            for i, o in enumerate(obj.os):
                os[i] = self*o
            return operatorsum(os)

        if isinstance(obj, statesum):
            sum = self*obj.states[0]
            for i in range(1,len(obj.states)):
                sum += self*obj.states[i]
            return sum
        
        if isinstance(obj, (int, float, complex)):
            sclr = scalar(obj, f'{obj}')
            return operatorproduct([sclr] + self._op)
        
        raise NotImplementedError
    
    def __rmul__(self, obj):

        if isinstance(obj, (int, float, complex)):
            sclr = scalar(obj, f'{obj}')
            return operatorproduct([sclr] + self._op)
        
        raise NotImplementedError
        
    def __eq__(self, obj):
        raise NotImplementedError
    
    def __str__(self):
        return self.string
    
    def __format__(self, spec):
        return self.string
    
    @property
    def dag(self):
        listdag = [0 for o in self._op]
        for i, o in enumerate(self._op):
            listdag[i] = o.dag
        listdag = list(reversed(listdag))
        return operatorproduct(listdag)
    
    @property
    def string(self):
        '''
        Return the string representing the product of operators.
        '''
        string = ''
        for i in range(len(self._op)):
            string += self._op[i].string
        return string
    
    @property
    def op(self):
        '''
        Return the list of multiplied operators
        '''
        return self._op
    
    @op.setter
    def op(self, operatorlist):
        self._op = operatorlist
    
class number(operator):
    ''' 
    Create a number operator.

    Parameters
    ----------
    indices : iterable or int
        The indices which the operator acts on. If `indices` is an `int` then it is assumed 
        to refer to the effective site.
    '''

    def __init__(self, indices):
        self._indices = indices
        self._calc_effsite()

    def __pos__(self):
        return self

    def __neg__(self):
        negone = scalar(-1, '(-1)')
        return operatorproduct([negone,self])

    def _count(self,bs):
        """ 
        Count the number of particles in a Fock state.

        :param bs: a bosonic Fock state
        :type bs: fockstate
        """
        
        if isinstance(bs,vacuum):
            return null()
        
        if bs.is_occupied(self._site): # if the site occupation number n_i != 0
            newamp = bs.amp * bs.state[self._site]
            return fockstate(bs.state, amp = newamp)
        else:
            return null()
        
    def _action(self,bs):
        return self._count(bs)
    
    def __str__(self):
        return self.string
    
    def __format__(self,spec): 
        return self.string
    
    def string(self):

        if isinstance(self._indices, int):
            string = str(self._indices)
        else:
            string = ''
            for i in range(len(self._indices)-1):
                string += str(self._indices[i]) + ','
            string += str(self._indices[-1])    
        return 'n' + _subscripts(string)
    
    @property
    def dag(self):
        '''
        Return the Hermitian adjoint of the number operator.
        '''
        return self
    
    @property
    def string(self):
        '''
        Return the string representing the number operator.
        '''
        if isinstance(self._indices, int):
            string = str(self._indices)
        else:
            string = ''
            for i in range(len(self._indices)-1):
                string += str(self._indices[i]) + ','
            string += str(self._indices[-1])    
        return 'n' + _subscripts(string)
    
class hamiltonian:
    ''' Create a hamiltonian.
        
        Parameters
        ----------
        obj : operator, operatorsum, operatorproduct or hamiltonian
            Operator form of the hamiltonian
    '''

    def __init__(self, obj):

        if isinstance(obj, hamiltonian):
            self._opform = obj.opform
            self.dense_matrix = obj.dense_matrix
            self.sparse_matrix = obj.sparse_matrix
        else:
            self._opform = obj
            self.dense_matrix = None
            self.sparse_matrix = None

    def __add__(self, obj):

        raise NotImplementedError
    
    def __mul__(self, obj):

        raise NotImplementedError

    def calc_matrix(self):
        '''
        Calculate the matrix form of the Hamiltonian.
        '''
        from numpy import zeros, csingle
        from scipy.sparse import csr_array as csr

        HM_upper = csr((_Nbasis, _Nbasis), dtype = csingle) # upper triangular sparse matrix
        idx = zeros(_Nbasis, dtype = int) # matrix elements of sparse matrix
        diag = zeros(_Nbasis, dtype = csingle) # matrix elements of sparse matrix
        jj = 0
        print(f'Calculating matrix elements of Hamiltonian.')
        for i in range(_Nbasis):
            row = zeros((_Nbasis-i), dtype=int) # row indices of sparse matrix
            column = zeros((_Nbasis-i), dtype=int) # column indices of sparse matrix
            data = zeros((_Nbasis-i), dtype = csingle) # matrix elements of sparse matrix
            ii = 0 # index of non-zero matrix elements in data array
            for j in range(i,_Nbasis):
                print(f'row: {i/_Nbasis*100:.2f}%, column: {(j-i)/(_Nbasis-i)*100:.2f}%', end = '\r')
                element = _fockspace.basis[i] * (self._opform * _fockspace.basis[j])
                if element != 0j:
                    if i == j:
                        idx[jj] = i
                        diag[jj] += element
                        jj += 1

                    row[ii] = i; column[ii] = j
                    data[ii] += element
                    ii += 1 
            HM_upper += csr((data,(row,column)), shape = (_Nbasis, _Nbasis), dtype = csingle) # add row to upper triangular matrix
        
        print(f'row: {100:.2f}%, column: {100:.2f}%')
        HM_diag = csr((diag, (idx,idx)), shape = (_Nbasis, _Nbasis), dtype = csingle)
        self.sparse_matrix = HM_upper + HM_upper.conj().T - HM_diag

    def eigsolve(self, sparse = True, numeigs = 2, which = 'SA'):
        '''
        Calculate the eigenvalues and eigenvectors of the Hamiltonian.

        Parameters
        ----------
        sparse : bool
            Determine whether to calculate the eigensystem using sparse or dense linear algebra.
        numeigs : int
            Number of eigenvalues and eigenvectors to return. Set to `'all'` to return all eigenvalues. 
        which : str
            Determine the order in which to calculate the eigenvalues. See `scipy.sparse.linalg.eigsh`.
        '''
        if numeigs == 'all':
            numeigs = _Nbasis

        if sparse is True:
            from scipy.sparse.linalg import eigsh
            self._eigvals, self._eigvecs = eigsh(self.sparse, k = numeigs, which = which)
        else:        
            from numpy.linalg import eigh
            self._eigvals, self._eigvecs = eigh(self.array)
            self._eigvals = self._eigvals[:, :numeigs]
            self._eigvecs = self._eigvecs[:, :numeigs]

    @property
    def array(self):
        '''
        Return the dense matrix form of the Hamiltonian. Calculates the matrix if
        it hasn't been already.
        :rtype: np.array
        '''
        if self.dense_matrix is None:
            if self.sparse_matrix is None:
                self.calc_matrix()
            self.dense_matrix = self.sparse_matrix.toarray()
        return self.dense_matrix
    
    @property
    def sparse(self):
        '''
        Return the sparse matrix form of the Hamiltonian. Calculates the matrix if
        it hasn't been already.
        :rtype: scipy.sparse.csr_matrix
        '''
        if self.sparse_matrix is None:
            self.calc_matrix()
        return self.sparse_matrix
    
    @property
    def eigvals(self):
        '''
        Return the calculated eigvals of the Hamiltonian.
        '''
        if self._eigvals is None:
            raise AttributeError('The eigensolutions must be calculated first')
        return self._eigvals
    
    @property
    def eigvecs(self):
        '''
        Return the calculated eigvecs of the Hamiltonian.
        '''
        if self._eigvecs is None:
            raise AttributeError('The eigensolutions must be calculated first')
        return self._eigvecs
    