# Assumptions:
# - The wavelengths and corresponding spectral lines are strictly increasing
# - The measurement error is normally distributed and homoscedastic
# - The model is linear in the parameters
using CairoMakie, Combinatorics, StatsBase
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

 
    factor = 1 / (length(spectral_lines) * length(wavelengths)) # extra weight to maximize hits (doesn't work?)
    MSE = factor * calculate_MSE(coeffs, spectral_lines, wavelengths) / var(wavelengths)
 
    return MSE, coeffs, cov_matrix
 
end
 
# Generate all possible strictly increasing subsequences of wavelengths
# with length equal to length(spectral_lines)
n_spectral = length(spectral_lines)
n_wavelengths = length(wavelengths)
 
#combination_count = binomial(n_wavelengths, n_spectral) * binomial(n_spectral, n_spectral-2)
#println("Number of combinations to evaluate: ", combination_count)
 
combination_count = 0
skip = 0
 
global best_MSE = Inf
global best_coeffs = nothing
global best_subset = nothing
global best_subset_sp = nothing
 
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
            skip += 1
            continue
        end
        subset_sp = spectral_lines[subset_indices_sp][1:min(end, length(subset_wavelengths))]
        subset_wavelengths = subset_wavelengths[1:length(subset_sp)]
        # Fit the linear model for this subset
        MSE, coeffs, cov_matrix = fit_linear_model(subset_sp, subset_wavelengths)
 
        # Check if this is the best fit so far
        if MSE < best_MSE
            global best_MSE = MSE
            global best_coeffs = coeffs
            global best_subset = subset_indices
            global best_subset_sp = subset_indices_sp
            global best_cov_matrix = cov_matrix
        end
        combination_count += 1 
    end
end
 
end_time = time()
runtime = end_time - start_time
 
println("Optimization completed!")
println("Runtime: ", round(runtime, digits=4), " seconds")
println("Combinations: ", combination_count)
println("Skipped: ", skip)
println("Average time per combination: ", round(runtime / combination_count * 1000, digits=4), " ms")
println("Best MSE: ", best_MSE)
println("Best coefficients: ", best_coeffs)
println("Best wavelength subset indices: ", best_subset)
println("Best wavelength subset: ", wavelengths[best_subset])
println("Best spectral subset indices: ", best_subset_sp)
println("Best spectral subset: ", spectral_lines[best_subset_sp])

# Create the plot
fig = Figure()
ax = Axis(fig[1, 1], xlabel="Spectral Lines", ylabel="Wavelengths",
    title="Linear Fit: Wavelengths vs Spectral Lines")
 
# Add scatter plot for data points
scatter!(ax, 
         spectral_lines[best_subset_sp][1:min(end, length(best_subset))], 
         wavelengths[best_subset][1:min(end, length(best_subset_sp))], label="Data points")
 
# Generate points for the fitted line
x_fit = range(minimum(spectral_lines), maximum(spectral_lines), length=100)
y_fit = predict(best_coeffs, x_fit)
 
# Add the fitted line
lines!(ax, x_fit, y_fit, label="Linear fit", linewidth=2)
 
#scatter!(ax, spectral_lines, best_coeffs[2]*spectral_lines .+ best_coeffs[1])
#scatter!(ax, (wavelengths .- best_coeffs[1])/best_coeffs[2], wavelengths)
# Add legend
axislegend(ax)
 
# Display the figure
fig
 
# reference:
#{22.181795919847822: 534.109,
# 41.140395614525275: 534.328,
# 144.26273501671918: 535.516,
# 186.4979599536216: 536.001,
# 292.1601184580567: 537.231,
# 377.61005873042376: 538.325}
 
# reference dispersion:
# 0.01156389139190282