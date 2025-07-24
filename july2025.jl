# Assumptions:
# - The wavelengths and corresponding spectral lines are strictly increasing
# - The measurement error is normally distributed and homoscedastic
# - The model is linear
using CairoMakie, Combinatorics, StatsBase, CSV, DataFrames
using ProgressMeter: Progress, next!

OPTIMIZE = false

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

function fit_linear_model(X, y, wavelengths)
    # Fit a linear model to the data
    # Calculate coefficients using a numerically stable method
    coeffs = X \ wavelengths
    # Calculate covariance matrix
    cov_matrix = inv(X' * X) * var(y)
    var_b = cov_matrix[2, 2]  # Variance of the slope

    # Calculate predictions and MSE
    predictions = X * coeffs
    MSE = calculate_MSE(predictions, wavelengths)

    return MSE, coeffs, var_b
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

function optimize_wavelength_fit(spectral_lines, wavelengths, lb_frac=0.7, n_best = 10)
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
    MIN_LINES = ceil(lb_frac * n_spectral)

    combination_count = 0
    for n_lines in MIN_LINES:n_spectral
        n_lines = Int(n_lines)
        combination_count += binomial(n_wavelengths, n_lines) * binomial(n_spectral, n_lines)
    end
    println("Number of combinations to evaluate: ", combination_count)

    # Store all results
    # Initialize arrays to store results efficiently
    results = Vector{NamedTuple{(:MSE, :varb, :coeffs, :subset_indices, :subset_indices_sp), 
                               Tuple{Float64, Float64, Vector{Float64}, Vector{Int}, Vector{Int}}}}()
    
    # Time the optimization process
    start_time = time()

    
    prog = Progress(combination_count; dt=0.05, desc="Optimizing... ", showspeed=true, color=:firebrick)

    for n_lines in  MIN_LINES:n_spectral
        n_lines = Int(n_lines)
        for subset_indices_sp in combinations(1:n_spectral, n_lines)
            # Extract the subsequence of spectral lines
            subset_sp = spectral_lines[subset_indices_sp]
            # Pre-calculate the design matrix as it is constant
            X_design = hcat(ones(n_lines), subset_sp) # Design matrix with intercept

            # Generate all combinations of wavelengths of size n_lines
            for subset_indices in combinations(1:n_wavelengths, n_lines)
                # Extract the subsequence of wavelengths and spectral lines
                subset_wavelengths = wavelengths[subset_indices]

                # Fit the linear model for this subset using the pre-calculated design matrix
                MSE, coeffs, varb = fit_linear_model(X_design, subset_sp, subset_wavelengths)

                # Store the results
                push!(results, (MSE=MSE, varb=varb, coeffs=coeffs, subset_indices=subset_indices, subset_indices_sp=subset_indices_sp))

                next!(prog)
            end
        end
    end
    
    # End timing
    end_time = time()
    runtime = end_time - start_time

    println("Optimization completed in $(runtime) seconds.")
    println("Total combinations evaluated: ", combination_count)
    # Create combined ranking metric (lower is better for both MSE and varb)
    # Normalize both metrics to [0,1] range and combine them
    mse_values = [r.MSE for r in results]
    varb_values = [r.varb for r in results]
    
    mse_min, mse_max = extrema(mse_values)
    varb_min, varb_max = extrema(varb_values)
    
    # Normalize and create combined score
    combined_scores = Float64[]
    for r in results
        normalized_mse = (r.MSE - mse_min) / (mse_max - mse_min)
        normalized_varb = (r.varb - varb_min) / (varb_max - varb_min)
        combined_score = normalized_mse + normalized_varb  # Equal weighting
        push!(combined_scores, combined_score)
    end
    
    # Sort results by combined score and keep top n_best
    sorted_indices = sortperm(combined_scores)
    best_results = results[sorted_indices[1:min(n_best, length(results))]]
    
    return best_results
end

function plot_n_best(best_results)
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
    
    n_best = size(best_results, 1)

    fig = Figure(size=(800, 600))
    ax = Axis(fig[1, 1], xlabel="Spectral Lines", ylabel="Wavelengths",
        title="Top $n_best Fits Based on Combined Ranking")


    # Plot reference data first
    ref_spectral = collect(keys(reference_data))
    ref_wavelengths = collect(values(reference_data))
    scatter!(ax, ref_spectral, ref_wavelengths,
        color=:black, marker=:cross, markersize=12,
        label="Reference Data")

    for (i, row) in enumerate(best_results)
        subset_sp = spectral_lines[row.subset_indices_sp]
        subset_wl = wavelengths[row.subset_indices]
        # # Trim to matching lengths, should not be necessary
        # min_len = min(length(subset_sp), length(subset_wl))
        # subset_sp = subset_sp[1:min_len]
        # subset_wl = subset_wl[1:min_len]

        # Plot data points
        scatter!(ax, subset_sp, subset_wl,
            color=Makie.wong_colors()[(i-1)%7+1], alpha=0.7, markersize=8,
            label="Rank $i (MSE: $(round(row.MSE, digits=3)))")

        # Generate fitted line
        x_range = range(minimum(subset_sp), maximum(subset_sp), length=50)
        y_fit = predict(row.coeffs, x_range)
        lines!(ax, x_range, y_fit, color=Makie.wong_colors()[(i-1)%7+1], linewidth=2)
    end

    axislegend(ax, position=:lt)
    fig
end

if OPTIMIZE
    # Main execution
    best_results = optimize_wavelength_fit(spectral_lines, wavelengths)

    # Save best results to a file
    # Convert results to DataFrame and save as CSV
    df = DataFrame(
        rank = 1:length(best_results),
        MSE = [r.MSE for r in best_results],
        varb = [r.varb for r in best_results],
        intercept = [r.coeffs[1] for r in best_results],
        slope = [r.coeffs[2] for r in best_results],
        subset_indices = [join(r.subset_indices, " ") for r in best_results],
        subset_indices_sp = [join(r.subset_indices_sp, " ") for r in best_results]
    )

    CSV.write("best_results.csv", df)
    println("Best results saved to best_results.csv")

    plot_n_best(best_results)
else 
    # Load pre-computed best results from file
    df = CSV.read("best_results.csv", DataFrame)
    best_results = []
    for i in 1:size(df, 1)
        push!(best_results, (
            MSE = df.MSE[i],
            varb = df.varb[i],
            coeffs = [df.intercept[i], df.slope[i]],
            subset_indices = parse.(Int, split(df.subset_indices[i], " ")),
            subset_indices_sp = parse.(Int, split(df.subset_indices_sp[i], " "))
        ))
    end
    # Plot the best n fits based on overall rank
    plot_n_best(best_results)
end