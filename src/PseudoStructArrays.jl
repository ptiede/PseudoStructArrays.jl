module PseudoStructArrays

export PseudoStructArray, fieldview

const PRIMITIVES = Union{
    Bool, Int8, Int16, Int32, Int64, Int128,
    UInt8, UInt16, UInt32, UInt64, UInt128,
    Float16, Float32, Float64,
}

"""
    nfields_static(::Type{T}) -> Int

Return the number of fields in type T as a compile-time constant.
"""
@inline nfields_static(::Type{T}) where {T} = fieldcount(T)

"""
    fieldtype_homogeneous(::Type{T}) -> Type

Check that all fields of T have the same type and return that type.
Throws an error if fields are not homogeneous.
"""
@generated function fieldtype_homogeneous(::Type{T}) where {T}
    T <: PRIMITIVES && return :($T)
    nf = fieldcount(T)
    nf == 0 && error("Type $T has no fields")
    ft = fieldtype(T, 1)
    for i in 2:nf
        fieldtype(T, i) === ft || error("All fields of $T must have the same type. Field 1 has type $ft, field $i has type $(fieldtype(T, i))")
    end
    return :($ft)
end

"""
    construct_from_tuple(::Type{T}, tup::NTuple{N}) -> T

Construct an instance of T from a tuple of field values.
"""
@generated function construct_from_tuple(::Type{T}, tup::NTuple{N, S}) where {T, N, S}
    nf = fieldcount(T)
    nf == N || error("Tuple length $N does not match field count $nf for type $T")
    args = [:(tup[$i]) for i in 1:nf]
    return :(T($(args...)))
end

"""
    destruct_to_tuple(x::T) -> NTuple{N, S}

Extract all fields of x into a tuple.
"""
@generated function destruct_to_tuple(x::T) where {T}
    nf = fieldcount(T)
    nf == 0 && return :(())
    ft = fieldtype(T, 1)
    args = [:(getfield(x, $i)) for i in 1:nf]
    return :(tuple($(args...))::NTuple{$nf, $ft})
end

"""
    PseudoStructArray{T, N, A, C} <: AbstractArray{T, N}

An array of isbits struct elements backed by a contiguous `(N+1)`-dimensional
array where the last dimension corresponds to the struct fields.

The struct type `T` must:
- Be an isbits type
- Have all fields of the same scalar type

# Type Parameters
- `T`: The element type (any suitable isbits struct)
- `N`: The number of dimensions of the PseudoStructArray
- `A`: The underlying array type (must be `AbstractArray{fieldtype, N+1}`)
- `C`: The NamedTuple type mapping field names to indices

# Examples
```julia
struct Point3D
    x::Float64
    y::Float64
    z::Float64
end

data = reshape(1.0:12.0, 4, 3)  # 4 elements with 3 fields each
psa = PseudoStructArray{Point3D}(data)
psa[1]  # Point3D(1.0, 5.0, 9.0)
```
"""
struct PseudoStructArray{T, N, A <: AbstractArray, C <: NamedTuple} <: AbstractArray{T, N}
    parent::A
    fieldmap::C  # Maps field names to indices: (x=1, y=2, z=3)

    function PseudoStructArray{T}(parent::A) where {T, A <: AbstractArray}
        # If T is a UnionAll (e.g., Point2D instead of Point2D{Float64}),
        # try to concretize it from the array element type
        T <: PRIMITIVES && return parent


        if T isa UnionAll
            S = eltype(parent)
            ConcreteT = _concretize_eltype(T, S)
            return PseudoStructArray{ConcreteT}(parent)
        end

        isbitstype(T) || error("Element type $T must be isbits")
        NF = nfields_static(T)
        FT = fieldtype_homogeneous(T)
        N = ndims(parent) - 1
        N >= 0 || error("Parent array must have at least 1 dimension")
        eltype(parent) === FT || error("Element type of parent array ($(eltype(parent))) must match field type ($FT)")
        size(parent, ndims(parent)) == NF || error("Last dimension size ($(size(parent, ndims(parent)))) must match number of fields ($NF)")

        # Store field name -> index mapping in a NamedTuple
        names = fieldnames(T)
        indices = ntuple(identity, Val(NF))
        fieldmap = NamedTuple{names}(indices)

        return new{T, N, A, typeof(fieldmap)}(parent, fieldmap)
    end
end

"""
    _concretize_eltype(::Type{T}, ::Type{S}) -> Type

For a parametric struct type T{P} where fields are of type P,
return the concrete type T{S}. For non-parametric types, returns T.
"""
function _concretize_eltype(::Type{T}, ::Type{S}) where {T, S}
    if T isa DataType && T.name.wrapper !== T
        # Already a concrete type
        return T
    elseif T isa UnionAll
        # Parametric type - try to concretize with S
        concrete = T{S}
        # Verify the fields actually use this type parameter
        if fieldcount(concrete) > 0 && fieldtype(concrete, 1) === S
            return concrete
        else
            error("Cannot automatically concretize $T with element type $S")
        end
    else
        return T
    end
end

# Get the parent array
Base.parent(psa::PseudoStructArray) = getfield(psa, :parent)

# Compile-time field info accessors
@inline _nfields(::PseudoStructArray{T}) where {T} = nfields_static(T)
@inline _nfields(::Type{<:PseudoStructArray{T}}) where {T} = nfields_static(T)
@inline _fieldtype(::Type{<:PseudoStructArray{T}}) where {T} = fieldtype_homogeneous(T)

# AbstractArray interface
Base.size(psa::PseudoStructArray) = Base.front(size(parent(psa)))
Base.length(psa::PseudoStructArray) = prod(size(psa))

# Linear indexing - get single element
Base.@propagate_inbounds function Base.getindex(psa::PseudoStructArray{T, N}, i::Int) where {T, N}
    @boundscheck checkbounds(psa, i)
    parent_arr = parent(psa)
    stride = length(psa)
    @inbounds return construct_from_tuple(T, ntuple(n -> parent_arr[i + (n - 1) * stride], _nfields(psa)))
end

# Cartesian indexing for N-dimensional access
Base.@propagate_inbounds function Base.getindex(psa::PseudoStructArray{T, N}, I::Vararg{Int, N}) where {T, N}
    @boundscheck checkbounds(psa, I...)
    parent_arr = parent(psa)
    @inbounds return construct_from_tuple(T, ntuple(n -> parent_arr[I..., n], _nfields(psa)))
end

# Setindex for linear indexing
Base.@propagate_inbounds function Base.setindex!(psa::PseudoStructArray{T, N}, v::T, i::Int) where {T, N}
    @boundscheck checkbounds(psa, i)
    parent_arr = parent(psa)
    stride = length(psa)
    tup = destruct_to_tuple(v)
    @inbounds for n in 1:_nfields(psa)
        parent_arr[i + (n - 1) * stride] = tup[n]
    end
    return v
end

# Setindex for Cartesian indexing
Base.@propagate_inbounds function Base.setindex!(psa::PseudoStructArray{T, N}, v::T, I::Vararg{Int, N}) where {T, N}
    @boundscheck checkbounds(psa, I...)
    parent_arr = parent(psa)
    tup = destruct_to_tuple(v)
    @inbounds for n in 1:_nfields(psa)
        parent_arr[I..., n] = tup[n]
    end
    return v
end

# Similar - create a new PseudoStructArray with the same structure
# For non-struct types (scalars like Float64), fall back to a regular Array
function Base.similar(psa::PseudoStructArray, ::Type{T2}, dims::Dims) where {T2}
    if _isnonemptystructtype(T2)
        NF = nfields_static(T2)
        FT = fieldtype_homogeneous(T2)
        new_parent = similar(parent(psa), FT, (dims..., NF))
        return PseudoStructArray{T2}(new_parent)
    else
        # For primitive types (Float64, Int, etc.), return a regular Array
        return similar(parent(psa), T2, dims)
    end
end

"""
    _isnonemptystructtype(::Type{T}) -> Bool

Check if type T is a struct type with at least one field.
This mirrors StructArrays.isnonemptystructtype.
"""
_isnonemptystructtype(::Type{T}) where {T} = isstructtype(T) && fieldcount(T) != 0

function Base.similar(psa::PseudoStructArray{T, N}) where {T, N}
    return similar(psa, T, size(psa))
end

# IndexStyle - forward from parent array
Base.IndexStyle(::Type{<:PseudoStructArray{T, N, A, C}}) where {T, N, A, C} = Base.IndexStyle(A)

# ============================================================================
# Broadcasting interface
# ============================================================================

struct PseudoStructArrayStyle{N} <: Broadcast.AbstractArrayStyle{N} end

# Style constructors
PseudoStructArrayStyle(::Val{N}) where {N} = PseudoStructArrayStyle{N}()
PseudoStructArrayStyle{M}(::Val{N}) where {M, N} = PseudoStructArrayStyle{N}()

# Register the broadcast style for PseudoStructArray
Base.BroadcastStyle(::Type{<:PseudoStructArray{T, N}}) where {T, N} = PseudoStructArrayStyle{N}()

# When combining with DefaultArrayStyle, PseudoStructArrayStyle wins (raises)
Base.BroadcastStyle(::PseudoStructArrayStyle{N}, ::Broadcast.DefaultArrayStyle{M}) where {N, M} =
    PseudoStructArrayStyle{max(N, M)}()
Base.BroadcastStyle(::Broadcast.DefaultArrayStyle{M}, ::PseudoStructArrayStyle{N}) where {N, M} =
    PseudoStructArrayStyle{max(N, M)}()

# similar for broadcasting - returns a regular Array since the result eltype
# is typically not a PseudoStructArray-compatible struct
function Base.similar(bc::Broadcast.Broadcasted{PseudoStructArrayStyle{N}}, ::Type{ElType}) where {N, ElType}
    return Array{ElType}(undef, axes(bc))
end

"""
    fieldview(psa::PseudoStructArray, field::Int)

Return a view into the `field`-th component of all elements in the array.
"""
@inline function fieldview(psa::PseudoStructArray{T, N}, field::Int) where {T, N}
    NF = nfields_static(T)
    @boundscheck 1 <= field <= NF || throw(BoundsError("field index $field out of range 1:$NF"))
    return selectdim(parent(psa), N + 1, field)
end

# Property access: psa.x returns a view into the x field
# getfield on the fieldmap NamedTuple gives us the index at compile time
@inline function Base.getproperty(psa::PseudoStructArray{T, N}, name::Symbol) where {T, N}
    idx = getfield(getfield(psa, :fieldmap), name)
    return selectdim(parent(psa), N + 1, idx)
end

function Base.propertynames(::PseudoStructArray{T}) where {T}
    return fieldnames(T)
end

# Pretty printing
function Base.show(io::IO, ::MIME"text/plain", psa::PseudoStructArray{T, N}) where {T, N}
    print(io, join(size(psa), "×"))
    print(io, " PseudoStructArray{$T, $N}")
    return if !isempty(psa)
        println(io, ":")
        Base.print_array(io, psa)
    end
end

end
