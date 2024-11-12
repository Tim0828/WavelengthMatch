# spectral_optimizer.jl
module SpectralOptimizer

export Optimizer, optimize_path

mutable struct Optimizer
    best_path::Vector{Float64}
    best_correlation::Float64
    current_path::Vector{Float64}
end

function _helper(self::Optimizer, 
                wldb::Vector{Float64}, 
                n::Int, 
                start::Int, 
                path_idx::Int,
                spl::Vector{Float64})
    if n == 0
        current_correlation = cor(view(self.current_path, 1:path_idx), spl)
        if current_correlation > self.best_correlation
            copyto!(self.best_path, 1, self.current_path, 1, path_idx)
            self.best_correlation = current_correlation
        end
        return
    end

    for i in start:length(wldb)
        if path_idx == 0 || wldb[i] > self.current_path[path_idx]
            self.current_path[path_idx + 1] = wldb[i]
            _helper(self, wldb, n-1, i+1, path_idx + 1, spl)
        end
    end
end

function julia_optimize_path(wavelengths::Vector{Float64}, 
                      pixels::Vector{Float64})::Tuple{Vector{Float64}, Float64}
    n = length(pixels)
    optimizer = Optimizer(wavelengths[1:n], 0.0, zeros(Float64, n))
    _helper(optimizer, wavelengths, n, 0, 0, pixels)
    return optimizer.best_path, optimizer.best_correlation
end

end

