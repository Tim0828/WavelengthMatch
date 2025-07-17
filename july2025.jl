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

function fit_linear_model(spectral_lines, wavelengths)
    # Fit a linear model to the data
    X = hcat(ones(length(spectral_lines)), spectral_lines)  # Design matrix with intercept
    y = wavelengths

    # Calculate coefficients using the normal equation
    coeffs = inv(X' * X) * X' * y

    MSE = calculate_MSE(coeffs, spectral_lines, wavelengths)

    return MSE, coeffs

end

# Generate all possible strictly increasing subsequences of wavelengths
# with length equal to length(spectral_lines)
n_spectral = length(spectral_lines)
n_wavelengths = length(wavelengths)

combination_count = binomial(n_wavelengths, n_spectral)
println("Number of combinations to evaluate: ", combination_count)

global best_MSE = Inf
global best_coeffs = nothing
global best_subset = nothing

# Time the optimization process
println("Starting optimization...")
start_time = time()

# Use combinations to generate all possible subsequences
for subset_indices in combinations(1:n_wavelengths, n_spectral)

    # Extract the subsequence of wavelengths
    subset_wavelengths = wavelengths[subset_indices]

    # Fit the linear model for this subset
    MSE, coeffs = fit_linear_model(spectral_lines, subset_wavelengths)

    # Check if this is the best fit so far
    if MSE < best_MSE
        global best_MSE = MSE
        global best_coeffs = coeffs
        global best_subset = subset_indices
    end
end

end_time = time()
runtime = end_time - start_time

println("Optimization completed!")
println("Runtime: ", round(runtime, digits=4), " seconds")
println("Average time per combination: ", round(runtime / combination_count * 1000, digits=4), " ms")
println("Best MSE: ", best_MSE)
println("Best coefficients: ", best_coeffs)
println("Best wavelength subset indices: ", best_subset)
println("Best wavelength subset: ", wavelengths[best_subset])



# Create the plot
fig = Figure()
ax = Axis(fig[1, 1], xlabel="Spectral Lines", ylabel="Wavelengths",
    title="Linear Fit: Wavelengths vs Spectral Lines")

# Add scatter plot for data points
scatter!(ax, spectral_lines, wavelengths[best_subset], label="Data points")

# Generate points for the fitted line
x_fit = range(minimum(spectral_lines), maximum(spectral_lines), length=100)
y_fit = predict(best_coeffs, x_fit)

# Add the fitted line
lines!(ax, x_fit, y_fit, label="Linear fit", linewidth=2)

# Add legend
axislegend(ax)

# Display the figure
fig

# # --- Backtracking Implementation ---
# function find_subsequences_backtrack(arr, n)
#     """
#     Generates all strictly increasing subsequences of length n using backtracking.
#     """
#     m = length(arr)
#     result = Vector{Vector{Float64}}()

#     function backtrack(start_index, current_subsequence)
#         # Base case: a valid subsequence of length n is found
#         if length(current_subsequence) == n
#             push!(result, copy(current_subsequence))
#             return
#         end

#         # Pruning: if remaining elements aren't enough
#         if n - length(current_subsequence) > m - start_index + 1
#             return
#         end

#         # Iterate to find the next element
#         for i in start_index:m
#             push!(current_subsequence, arr[i])
#             backtrack(i + 1, current_subsequence)
#             pop!(current_subsequence) # Backtrack
#         end
#     end

#     backtrack(1, Vector{Float64}())
#     return result
# end


# # --- Original Method: Combinatorics ---
# println("--- Combinatorics Method ---")
# println("Best MSE: ", best_MSE)
# println("Best coefficients: ", best_coeffs)
# println("Best wavelength subset indices: ", best_subset)
# println("Best wavelength subset: ", wavelengths[best_subset])


# # --- New Method: Backtracking ---
# println("\n--- Backtracking Method ---")
# best_MSE_backtrack = Inf
# best_coeffs_backtrack = nothing
# best_subset_backtrack = nothing

# println("Starting backtracking optimization...")
# start_time_backtrack = time()

# subsequences = find_subsequences_backtrack(wavelengths, n_spectral)
# combination_count_backtrack = length(subsequences)

# for subset_wavelengths in subsequences
#     MSE, coeffs = fit_linear_model(spectral_lines, subset_wavelengths)
#     if MSE < best_MSE_backtrack
#         best_MSE_backtrack = MSE
#         best_coeffs_backtrack = coeffs
#         # Note: We don't have indices here, just the subset itself
#     end
# end

# end_time_backtrack = time()
# runtime_backtrack = end_time_backtrack - start_time_backtrack

# println("Backtracking optimization completed!")
# println("Runtime: ", round(runtime_backtrack, digits=4), " seconds")
# println("Combinations evaluated: ", combination_count_backtrack)
# println("Average time per combination: ", round(runtime_backtrack / combination_count_backtrack * 1000, digits=4), " ms")
# println("Best MSE: ", best_MSE_backtrack)
# println("Best coefficients: ", best_coeffs_backtrack)



# # Create the plot (using results from the original, faster method)
# fig2 = Figure()
# ax2 = Axis(fig2[1, 1], xlabel="Spectral Lines", ylabel="Wavelengths",
#     title="Linear Fit: Wavelengths vs Spectral Lines (Backtracking)")

# # Add scatter plot for data points
# scatter!(ax2, spectral_lines, wavelengths[best_subset], label="Data points")

# # Generate points for the fitted line
# x_fit_backtrack = range(minimum(spectral_lines), maximum(spectral_lines), length=100)
# y_fit_backtrack = predict(best_coeffs_backtrack, x_fit_backtrack)

# # Add the fitted line
# lines!(ax2, x_fit_backtrack, y_fit_backtrack, label="Linear fit", linewidth=2)

# # Add legend
# axislegend(ax2)

# # Display the figure
# fig2