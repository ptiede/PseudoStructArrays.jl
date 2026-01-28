using PseudoStructArrays
using Test

# Test structs
struct Point3D
    x::Float64
    y::Float64
    z::Float64
end

struct Point2D{T}
    x::T
    y::T
end

struct RGBA
    r::Float32
    g::Float32
    b::Float32
    a::Float32
end

@testset "PseudoStructArrays.jl" begin

    @testset "1D PseudoStructArray - Basic" begin
        # Create a 4-element array with 3 fields each
        data = reshape(collect(1.0:12.0), 4, 3)
        psa = PseudoStructArray{Point3D}(data)

        @test size(psa) == (4,)
        @test length(psa) == 4
        @test eltype(psa) == Point3D

        # Test linear indexing
        @test psa[1] == Point3D(1.0, 5.0, 9.0)
        @test psa[2] == Point3D(2.0, 6.0, 10.0)
        @test psa[3] == Point3D(3.0, 7.0, 11.0)
        @test psa[4] == Point3D(4.0, 8.0, 12.0)

        # Test Cartesian indexing (1D)
        @test psa[1] == psa[CartesianIndex(1)]
    end

    @testset "1D PseudoStructArray - setindex!" begin
        data = reshape(collect(1.0:12.0), 4, 3)
        psa = PseudoStructArray{Point3D}(data)

        # Test setindex with struct
        psa[1] = Point3D(100.0, 200.0, 300.0)
        @test psa[1] == Point3D(100.0, 200.0, 300.0)

        # Verify underlying data was modified
        @test data[1, 1] == 100.0
        @test data[1, 2] == 200.0
        @test data[1, 3] == 300.0
    end

    @testset "2D PseudoStructArray" begin
        # Create a 3x4 array with 2 fields each
        data = reshape(collect(1.0:24.0), 3, 4, 2)
        psa = PseudoStructArray{Point2D{Float64}}(data)

        @test size(psa) == (3, 4)
        @test length(psa) == 12
        @test eltype(psa) == Point2D{Float64}

        # Test Cartesian indexing
        @test psa[1, 1] == Point2D(1.0, 13.0)
        @test psa[2, 1] == Point2D(2.0, 14.0)
        @test psa[1, 2] == Point2D(4.0, 16.0)
        @test psa[3, 4] == Point2D(12.0, 24.0)

        # Test linear indexing on 2D array
        @test psa[1] == Point2D(1.0, 13.0)

        # Test setindex with Cartesian
        psa[2, 3] = Point2D{Float64}(100.0, 200.0)
        @test psa[2, 3] == Point2D(100.0, 200.0)
    end

    @testset "Parametric struct - type inference" begin
        # Test that PseudoStructArray{Point2D} infers Point2D{Float64}
        data = reshape(collect(1.0:8.0), 4, 2)
        psa = PseudoStructArray{Point2D}(data)

        @test eltype(psa) == Point2D{Float64}
        @test psa[1] == Point2D(1.0, 5.0)

        # Test with Float32
        data32 = reshape(collect(Float32, 1:8), 4, 2)
        psa32 = PseudoStructArray{Point2D}(data32)

        @test eltype(psa32) == Point2D{Float32}
        @test psa32[1] == Point2D(1.0f0, 5.0f0)
    end

    @testset "Explicit concrete type" begin
        data = reshape(collect(1.0:8.0), 4, 2)
        psa = PseudoStructArray{Point2D{Float64}}(data)

        @test eltype(psa) == Point2D{Float64}
        @test psa[1] == Point2D(1.0, 5.0)
    end

    @testset "4-field struct (RGBA)" begin
        data = reshape(collect(Float32, 1:16), 4, 4)
        psa = PseudoStructArray{RGBA}(data)

        @test size(psa) == (4,)
        @test psa[1] == RGBA(1.0f0, 5.0f0, 9.0f0, 13.0f0)
        @test psa[4] == RGBA(4.0f0, 8.0f0, 12.0f0, 16.0f0)
    end

    @testset "fieldview" begin
        data = reshape(collect(1.0:12.0), 4, 3)
        psa = PseudoStructArray{Point3D}(data)

        fv1 = fieldview(psa, 1)
        fv2 = fieldview(psa, 2)
        fv3 = fieldview(psa, 3)

        @test fv1 == [1.0, 2.0, 3.0, 4.0]
        @test fv2 == [5.0, 6.0, 7.0, 8.0]
        @test fv3 == [9.0, 10.0, 11.0, 12.0]

        # fieldview should be a view, not a copy
        fv1[1] = 100.0
        @test psa[1].x == 100.0
    end

    @testset "similar" begin
        data = reshape(collect(1.0:12.0), 4, 3)
        psa = PseudoStructArray{Point3D}(data)

        psa2 = similar(psa)
        @test size(psa2) == size(psa)
        @test eltype(psa2) == eltype(psa)

        psa3 = similar(psa, Point3D, (2, 3))
        @test size(psa3) == (2, 3)
        @test eltype(psa3) == Point3D
    end

    @testset "Type stability and isbits" begin
        data = reshape(collect(1.0:12.0), 4, 3)
        psa = PseudoStructArray{Point3D}(data)

        # The element type should be isbits
        @test isbits(psa[1])

        # Check type inference for 1D case
        @inferred getindex(psa, 1)
        @inferred setindex!(psa, Point3D(1.0, 2.0, 3.0), 1)
        @inferred getproperty(psa, :x)
        @inferred size(psa)
        @inferred length(psa)
        @inferred fieldview(psa, 1)

        # 2D case
        data2d = reshape(collect(1.0:24.0), 3, 4, 2)
        psa2d = PseudoStructArray{Point2D{Float64}}(data2d)
        @inferred getindex(psa2d, 1, 1)
        @inferred getindex(psa2d, 2, 3)
        @inferred setindex!(psa2d, Point2D(1.0, 2.0), 1, 1)
        @inferred size(psa2d)
        @inferred fieldview(psa2d, 1)
    end

    @testset "Bounds checking" begin
        data = reshape(collect(1.0:12.0), 4, 3)
        psa = PseudoStructArray{Point3D}(data)

        @test_throws BoundsError psa[0]
        @test_throws BoundsError psa[5]
        @test_throws BoundsError fieldview(psa, 0)
        @test_throws BoundsError fieldview(psa, 4)
    end

    @testset "Construction validation" begin
        # Wrong element type
        data_int = reshape(collect(1:12), 4, 3)
        @test_throws Exception PseudoStructArray{Point3D}(data_int)

        # Wrong number of fields
        data_wrong = reshape(collect(1.0:8.0), 4, 2)
        @test_throws Exception PseudoStructArray{Point3D}(data_wrong)
    end

    @testset "Iteration and collection" begin
        data = reshape(collect(1.0:12.0), 4, 3)
        psa = PseudoStructArray{Point3D}(data)

        # Should be iterable
        collected = collect(psa)
        @test length(collected) == 4
        @test all(x -> x isa Point3D, collected)
        @test collected[1] == Point3D(1.0, 5.0, 9.0)

        # For loop
        count = 0
        for p in psa
            count += 1
            @test p isa Point3D
        end
        @test count == 4
    end

    @testset "Broadcasting" begin
        data = reshape(collect(1.0:12.0), 4, 3)
        psa = PseudoStructArray{Point3D}(data)

        # Map over elements
        xs = map(p -> p.x, psa)
        @test xs == [1.0, 2.0, 3.0, 4.0]
    end

    @testset "Parent array access" begin
        data = reshape(collect(1.0:12.0), 4, 3)
        psa = PseudoStructArray{Point3D}(data)

        @test PseudoStructArrays.parent(psa) === data
    end
end
