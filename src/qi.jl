module qi
using LinearAlgebra
using SparseArrays
using QuantumInformation
using Dates
#############################################
function id(d::Int64)
	return Matrix{Float64}(I,d,d)
end
#############################################
function kronn(var...)
	a = var[1]
	for i = 2:length(var)
		a = kron(a,var[i])
	end
	return a
end
#############################################
function pauli(s::Int64,n::Int64=1,j::Int64=1)
	# Create a Pauli Matrice with s = x,y,z = 1,2,3 on site j out of n qubits
	if n ==1
		if s==0
			return [1.0 0.0;0.0 1.0]
		elseif s==1
			return [0.0 1.0;1.0 0.0]
		elseif s==2
			return [0.0 -1.0*im;1.0*im 0.0]
		elseif s==3
			return [1.0 0.0;0.0 -1.0]
		else 
			return [0.0 0.0;0.0 0.0]
		end
	elseif n==0
		return [1.0]
	else
		return kronn(id(2^(j-1)),pauli(s),id(2^(n-j)))
	end
end
#############################################

#############################################
# function randU(n)
#     Q, R = qr(randn(ComplexF64, n, n))
#     Q * Diagonal(R./abs.(R))
# end
#############################################
function vonNeumannEntropy(rho::Array{Complex{Float64},2})
	eigs = eigvals(rho)
	s = 0.0
	# println(eigs)
	for val in eigs
		# println(val)
		if val>0.0
			s += - val*log(val)
		end
	end
	return s
end
#############################################
function spId(d::Int64)
	# return a dxd sparse identity matrix
	return sparse(collect(1:d),collect(1:d),ones(ComplexF64,d))
end
#############################################
function spDiagonal(diag::Array{Complex{Float64},1})
	# Create a sparse diagonal matrix
	d = length(diag)
	return sparse(collect(1:d),collect(1:d),diag)
end
#############################################
function spPauli(s::Int64,n::Int64=1,j::Int64=1)::SparseMatrixCSC{ComplexF64,Int64}
	# Create a Pauli Matrice with s = x,y,z = 1,2,3 on site j out of n qubits
	if n ==1
		if s==0
			return sparse([1.0 0.0;0.0 1.0])
		elseif s==1
			return sparse([0.0 1.0;1.0 0.0])
		elseif s==2
			return sparse([0.0 -1.0*im;1.0*im 0.0])
		elseif s==3
			return sparse([1.0 0.0;0.0 -1.0])
		else 
			return sprase([0.0 0.0;0.0 0.0])
		end
	elseif n==0
		return sprase([1.0])
	else
		return kronn(spId(2^(j-1)),spPauli(s),spId(2^(n-j)))
	end
end
#############################################
function PauliPauliNearestNeighbor(n::Int64,s::Int64)::SparseMatrixCSC{ComplexF64,Int64}
	# sum of nearest neighbor Pauli_s Pauli_s interaction in 1D chain of length n 
	HX = spzeros(2^n,2^n)
	for i = 1:n-1
		HX = HX + spPauli(s,n,i)*spPauli(s,n,i+1)
	end
	return HX
end
#############################################
function diagForm(HTerms::Tuple{Vararg{Union{SparseMatrixCSC{ComplexF64,Int64},Array{Complex{Float64},2}}}})
	return tuple([eigen(Matrix(term)) for term in HTerms]...)
end
#############################################
function randU(d::Int64)
	# Random dxd unitary drawn from the Haar measure
	return rand(CUE(d))
end
#############################################
function kronpower(U::Union{Matrix,SparseMatrixCSC},n::Int64)
	S = U
	for i=2:n
		S = kron(S,U)
	end
	return S
end
#############################################
function timestamp()
	return string(today())*"-"*string(Int(round(rand()*1000)))
end
#############################################
function printvar(file::IOStream,var)
	try
		if ndims(var)==1
			print(file,var[1])
			for i=2:length(var) 
				print(file,",",var[i])
			end
			println(file,"")
		end
		if ndims(var)==2
			for j =1:size(var,1)
				print(file,var[j,1])
				for i=2:size(var,2) 
					print(file,",",var[j,i])
				end
				println(file,"")
			end
		end
	catch
		println(file,var)
	end
end
#############################################
function save2file(filename::String,vars...)
	open(filename,"w") do file
   		for var in vars
   			printvar(file,var)
   		end
	end
end
#############################################
export pauli
export id
export kronn
export vonNeumannEntropy
export spId
export spDiagonal
export spPauli
export PauliPauliNearestNeighbor
export diagForm
export randU
export kronpower
export timestamp
export save2file
# export randU
end # module
