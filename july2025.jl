# Assumptions:
# - The wavelengths and corresponding spectral lines are strictly increasing
# - The measurement error is normally distributed and homoscedastic
# - The model is linear
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
    return coeffs[1] .+ coeffs[2] .* spectral_lines
end

function calculate_MSE(predictions, wavelengths)
    # Calculate Mean Squared Error
    return mean((predictions .- wavelengths) .^ 2)
end

function fit_linear_model(X, wavelengths)
    # Fit a linear model to the data
    # Calculate coefficients using a numerically stable method
    coeffs = X \ wavelengths

    # Calculate predictions and MSE
    predictions = X * coeffs
    MSE = calculate_MSE(predictions, wavelengths)

    return MSE, coeffs
end

function residuals(y, predicted)
    # Calculate residuals
    return y .- predicted   
end

function calculate_residual_standard_error(y, predicted)
    res = residuals(y, predicted)
    std_res = std(res)
    # Calculate standard error of the residuals (99% confidence interval)
    se = 2.576 * std_res / sqrt(length(res))
    return se
end


function plot_linear_fit(spectral_lines, wavelengths, best_subset, best_coeffs)
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
    display(fig)

    return fig
end

function plot_residuals(residuals, se)
    # Create the plot for residuals
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="Index", ylabel="Residuals",
        title="Residuals of the Linear Fit")

    # Add scatter plot for residuals
    scatter!(ax, 1:length(residuals), residuals, label="Residuals")

    # Add horizontal line for confidence interval
    hlines!(ax, [se, -se], label="Confidence Interval", color=:red, linestyle=:dash)
    # Add legend
    axislegend(ax)

    # Display the figure
    display(fig)

    return fig
end

function optimize_wavelength_fit(spectral_lines, wavelengths)
    # Ensure spectral_lines and wavelengths are strictly increasing
    if !issorted(spectral_lines) || !issorted(wavelengths)
        error("Spectral lines and wavelengths must be strictly increasing.")
    end

    # Ensure the lengths match
    if length(spectral_lines) > length(wavelengths)
        error("Spectral lines cannot exceed the number of wavelengths.")
    end

    # Generate all possible strictly increasing subsequences of wavelengths
    # with length equal to length(spectral_lines)
    n_spectral = length(spectral_lines)
    n_wavelengths = length(wavelengths)

    combination_count = binomial(n_wavelengths, n_spectral)
    println("Number of combinations to evaluate: ", combination_count)

    # Pre-calculate the design matrix as it is constant
    X_design = hcat(ones(n_spectral), spectral_lines)

    best_MSE = Inf
    best_coeffs = nothing
    best_subset = nothing

    # Time the optimization process
    println("Starting optimization...")
    start_time = time()

    # Use combinations to generate all possible subsequences
    for subset_indices in combinations(1:n_wavelengths, n_spectral)

        # Extract the subsequence of wavelengths
        subset_wavelengths = wavelengths[subset_indices]

        # Fit the linear model for this subset using the pre-calculated design matrix
        MSE, coeffs = fit_linear_model(X_design, subset_wavelengths)

        # Check if this is the best fit so far
        if MSE < best_MSE
            best_MSE = MSE
            best_coeffs = coeffs
            best_subset = subset_indices
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

    # plot the fit
    plot_linear_fit(spectral_lines, wavelengths, best_subset, best_coeffs)

    # Calculate residuals and confidence interval
    res = residuals(wavelengths[best_subset], predict(best_coeffs, spectral_lines))
    se = calculate_residual_standard_error(wavelengths[best_subset], predict(best_coeffs, spectral_lines))

    # Plot residuals
    plot_residuals(res, se)

    return best_MSE, best_coeffs, best_subset, res, se
end

# Run the optimization
best_MSE, best_coeffs, best_subset, res, se = optimize_wavelength_fit(spectral_lines, wavelengths)

# If any residuals are outside the confidence interval, print them and remove
if any(abs.(res) .> se)
    println("Residuals outside confidence interval: ", res[abs.(res) .> se])
    filtered_indices = findall(abs.(res) .<= se)
end

