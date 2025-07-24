# Assumptions:
# - The wavelengths and corresponding spectral lines are strictly increasing
# - The measurement error is normally distributed and homoscedastic
# - The model is linear in the parameters
using CairoMakie, Combinatorics, StatsBase

# Reference data
reference_data = Dict(
    22.181795919847822 => 534.109,
    41.140395614525275 => 534.328,
    144.26273501671918 => 535.516,
    186.4979599536216 => 536.001,
    292.1601184580567 => 537.231,
    377.61005873042376 => 538.325
)

# Reference dispersion
reference_dispersion = 0.01156389139190282

function read_data(spectral_file, wl_file)
    # Read spectral lines and wavelengths from files
    spectral_lines = readlines(spectral_file)
    wavelengths = readlines(wl_file)
 
    # Convert to Float64
    spectral_lines = parse.(Float64, spectral_lines)
    wavelengths = parse.(Float64, wavelengths)
 
    return spectral_lines, wavelengths
end
 
spectral_file = "spl.txt"
wl_file = "wldb.txt"
spectral_lines, wavelengths = read_data(spectral_file, wl_file)
println("Spectral Lines: ", spectral_lines)
println("Wavelengths: ", wavelengths)
 
function predict(coeffs, spectral_lines)
    # Predict wavelengths using the fitted model
    X = hcat(ones(length(spectral_lines)), spectral_lines)  # Design matrix with intercept
    return coeffs[1] .+ coeffs[2] .* spectral_lines
end
 
function calculate_MSE(coeffs, spectral_lines, wavelengths)
    predictions = predict(coeffs, spectral_lines)
    # Calculate Mean Squared Error
    return mean((predictions .- wavelengths) .^ 2)
end

 
test = 0
 
function fit_linear_model(spectral_lines, wavelengths)
    # Fit a linear model to the data
    X = hcat(ones(length(spectral_lines)), spectral_lines)  # Design matrix with intercept
    y = wavelengths
 
    # Calculate coefficients using the normal equation
    coeffs = inv(X' * X) * X' * y

    # Calculate covariance matrix
    cov_matrix = inv(X' * X) * var(y)
    var_b = cov_matrix[2, 2]  # Variance of the slope

 
    factor = 1 / (length(spectral_lines) * length(wavelengths)) # extra weight to maximize hits (doesn't work?)
    MSE = factor * calculate_MSE(coeffs, spectral_lines, wavelengths) / var(wavelengths)
 
    return MSE, coeffs, var_b
 
end
 
# Generate all possible strictly increasing subsequences of wavelengths
# with length equal to length(spectral_lines)
n_spectral = length(spectral_lines)
n_wavelengths = length(wavelengths)
 
#combination_count = binomial(n_wavelengths, n_spectral) * binomial(n_spectral, n_spectral-2)
#println("Number of combinations to evaluate: ", combination_count)
 
 
# Store all results
all_results = []
 
# Time the optimization process
println("Starting optimization...")
start_time = time()
 
MIN_LINES = 5
 
# Use combinations to generate all possible subsequences
for subset_indices in powerset(1:n_wavelengths, MIN_LINES, n_spectral)
 
    # Extract the subsequence of wavelengths
    subset_wavelengths = wavelengths[subset_indices]
 
    for subset_indices_sp in powerset(
            1:n_spectral, max(MIN_LINES, length(subset_wavelengths)-1), length(subset_wavelengths))
        if length(subset_wavelengths) > length(subset_indices_sp) + 1 #+ MIN_LINES
            
            continue
        end
        subset_sp = spectral_lines[subset_indices_sp][1:min(end, length(subset_wavelengths))]
        subset_wavelengths = subset_wavelengths[1:length(subset_sp)]
        # Fit the linear model for this subset
        MSE, coeffs, var_b = fit_linear_model(subset_sp, subset_wavelengths)
 
        # Store the result
        push!(all_results, (MSE=MSE, coeffs=coeffs, subset_indices=subset_indices,
                           subset_indices_sp=subset_indices_sp, var_b=var_b))

        
    end
end

# Add rankings based on MSE and var_b
mse_sorted = sort(all_results, by=x -> x.MSE)
var_b_sorted = sort(all_results, by=x -> x.var_b)

# Create ranking dictionaries
mse_ranks = Dict()
var_b_ranks = Dict()

for (i, result) in enumerate(mse_sorted)
    mse_ranks[result] = i
end

for (i, result) in enumerate(var_b_sorted)
    var_b_ranks[result] = i
end

# Add combined ranking (sum of individual ranks, lower is better)
for result in all_results
    mse_rank = mse_ranks[result]
    var_b_rank = var_b_ranks[result]
    combined_rank = mse_rank + var_b_rank
    
    # Add rankings to the result tuple
    result = merge(result, (mse_rank=mse_rank, var_b_rank=var_b_rank, combined_rank=combined_rank))
    
    # Update in the array
    idx = findfirst(x -> x.subset_indices == result.subset_indices && x.subset_indices_sp == result.subset_indices_sp, all_results)
    all_results[idx] = result
end

# Sort by combined rank to find overall best
sort!(all_results, by=x -> x.combined_rank)

 
end_time = time()
runtime = end_time - start_time
 
println("Optimization completed!")
println("Runtime: ", round(runtime, digits=4), " seconds")
 
# Plot the best n fits based on overall rank
n_best = min(5, length(all_results))  # Show top 5 or fewer if less available

fig = Figure(size=(800, 600))
ax = Axis(fig[1, 1], xlabel="Spectral Lines", ylabel="Wavelengths",
    title="Top $n_best Fits Based on Combined Ranking")

colors = [:red, :blue, :green, :orange, :purple]

# Plot reference data first
ref_spectral = collect(keys(reference_data))
ref_wavelengths = collect(values(reference_data))
scatter!(ax, ref_spectral, ref_wavelengths,
    color=:black, marker=:cross, markersize=12,
    label="Reference Data")

for i in 1:n_best
    result = all_results[i]
    subset_sp = spectral_lines[result.subset_indices_sp]
    subset_wl = wavelengths[result.subset_indices]
    
    # Trim to matching lengths
    min_len = min(length(subset_sp), length(subset_wl))
    subset_sp = subset_sp[1:min_len]
    subset_wl = subset_wl[1:min_len]
    
    # Plot data points
    scatter!(ax, subset_sp, subset_wl, 
             color=colors[i], alpha=0.7, markersize=8,
             label="Rank $i (MSE: $(round(result.MSE, digits=6)))")
    
    # Generate fitted line
    x_range = range(minimum(subset_sp), maximum(subset_sp), length=50)
    y_fit = predict(result.coeffs, x_range)
    lines!(ax, x_range, y_fit, color=colors[i], linewidth=2)
end

axislegend(ax, position=:lt)
fig