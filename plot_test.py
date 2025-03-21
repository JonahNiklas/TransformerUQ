import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ipywidgets as widgets
from IPython.display import display, clear_output

# Generate sample data
np.random.seed(0)
steps = np.arange(0, 100)
model_types = ['Model A', 'Model B', 'Model C']
acquisition_functions = ['BeamScore', 'BALD', 'BLEUVar', 'mpnet_cosine']
evaluation_functions = ['F1Eval', 'BLEU_eval', 'TargetUsageEval', 'ConceptUsageEval']

# Create more complex data structure with multiple dimensions
data = []
for model in model_types:
    for aq_func in acquisition_functions:
        for eval_func in evaluation_functions:
            # Create slightly different random patterns for each combination
            base = np.cumsum(np.random.randn(len(steps)))
            # Add some characteristic patterns based on the function types
            if aq_func == 'BALD':
                base += 2
            elif aq_func == 'BLEUVar':
                base -= 1
            
            if eval_func == 'BLEU_eval':
                base *= 1.2
            elif eval_func == 'ConceptUsageEval':
                base *= 0.8
                
            data.append(pd.DataFrame({
                'step': steps, 
                'value': base, 
                'model': model,
                'acquisition_function': aq_func,
                'evaluation_function': eval_func
            }))
            
df = pd.concat(data)

# Create widgets for filtering
model_selector = widgets.SelectMultiple(
    options=model_types,
    value=model_types,  # default: show all
    description='Models',
    disabled=False
)

acquisition_selector = widgets.SelectMultiple(
    options=acquisition_functions,
    value=[acquisition_functions[0]],  # default: first one
    description='Acquisition',
    disabled=False
)

evaluation_selector = widgets.SelectMultiple(
    options=evaluation_functions,
    value=[evaluation_functions[0]],  # default: first one
    description='Evaluation',
    disabled=False
)

# Add title input
title_input = widgets.Text(
    value='Benchmark Retention Curve',
    placeholder='Enter plot title',
    description='Title:',
    disabled=False
)

# Add export button
export_button = widgets.Button(
    description='Export to SVG',
    disabled=False,
    button_style='success',
    tooltip='Export current plot to SVG file'
)

output_filename = widgets.Text(
    value='plot_export.svg',
    placeholder='filename.svg',
    description='Filename:',
    disabled=False
)

# Create output area for the plot
plot_output = widgets.Output()

def create_plot(selected_models, selected_acquisitions, selected_evaluations, plot_title):
    fig = plt.figure(figsize=(10, 6))
    
    # Filter data based on all selections
    for model in selected_models:
        for aq_func in selected_acquisitions:
            for eval_func in selected_evaluations:
                filtered_df = df[(df['model'] == model) & 
                                 (df['acquisition_function'] == aq_func) &
                                 (df['evaluation_function'] == eval_func)]
                
                if not filtered_df.empty:
                    label = f"{model} - {aq_func} - {eval_func}"
                    plt.plot(filtered_df['step'], filtered_df['value'], label=label)
    
    plt.xlabel('Number of Samples')
    plt.ylabel('Evaluation Score')
    plt.title(plot_title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.7)
    return fig

def update_plot(selected_models, selected_acquisitions, selected_evaluations, plot_title):
    with plot_output:
        clear_output(wait=True)
        fig = create_plot(selected_models, selected_acquisitions, selected_evaluations, plot_title)
        plt.show()

# Export function
def export_svg(b):
    fig = create_plot(
        model_selector.value,
        acquisition_selector.value,
        evaluation_selector.value,
        title_input.value
    )
    filename = output_filename.value
    if not filename.endswith('.svg'):
        filename += '.svg'
    fig.savefig(filename, format='svg', bbox_inches='tight')
    plt.close(fig)
    print(f"Plot exported to {filename}")

# Connect the button to the export function
export_button.on_click(export_svg)

# Display the export controls
export_controls = widgets.HBox([output_filename, export_button])
display(export_controls)

# Use interactive to update the plot when any selection changes
interactive_plot = widgets.interactive(
    update_plot, 
    selected_models=model_selector,
    selected_acquisitions=acquisition_selector,
    selected_evaluations=evaluation_selector,
    plot_title=title_input
)

display(interactive_plot)
display(plot_output)