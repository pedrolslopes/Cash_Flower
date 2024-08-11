# Cash Flower: A Cashflow Simulator for Scientific Organizations

Cash Flower is a Python-based simulation tool designed to model various cash flow scenarios within scientific organizations, research groups, or institutes. It helps scientific team managers understand the financial impact of different cash flow events in their groups, including individual funding events, milestone-based projects, mid-term recurring payments, and stable sources of income or expenses. The simulator can generate key insights into potential risks, including the probability of running into a negative balance.

## Features

- **Simulation of Cash Flow Categories**:
  - **Stable**: Constant sources of income or expense (e.g., salaries, bills, IP licenses).
  - **Singles**: One-time revenue or expense events (e.g., rare large sales, purchases, grants).
  - **Recurrent**: Repeated cash flows triggered after an initial event (e.g., long-term deals, scholarships, sales or purchase of established products).
  - **Project**: Milestone-based cash flows tied to project success and timing.

- **Visualization**:
  - Monthly income and expenses histograms.
  - Monthly cash flow estimates with standard deviation bounds.
  - Accumulated funds over time, showing trends and potential risks.
  - Risk assessment of encountering a negative balance during the simulation period.
  - Analysis for a user-defined amount of months, given a pre-defined starting date of monitoring finances. Plots are adjusted to show status from the current date, plus the amount of months forecast is desired for.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/CashFlowSim.git
   cd CashFlowSim
   ```

2. Install the required Python packages:

    ```bash
    pip install numpy matplotlib
    ```
Both script and interactive versions of the code are available.

## Usage
1. **Define the Simulation Parameters:** In the script, configure the parameters for your cash flow events via dictionaries for the following four categories:

    - Singles: Define single-instance events like grants or large sales. Include the name of the event, the likelihood of it concretizing, the month (relative to your analysis starting point), and the amount of the funding. Amounts can be set as positive or negative.
    - Recurrent: Configure events that recur over time, such as scholarships or service contracts. Besides the above, this entry also takes how many months the recurrent cashflow event will last, and frequency of payments. Amounts can, again, be set as positive or negative.
    - Stable: Set up constant flows of income or expenses like salaries and utilities. These events are characterized by their names, average values, and standard deviations. Since amounts values of specific events are not stated explicitly, entries in this category also must be classified as incomes or expenses.
    - Projects: Add projects with milestone-based payments. Similar to the first two, but one can also include expected dates of deliverables tied to payments, and the chance missing a deliverable would kill the project.

2. **Run the Simulation:** Execute the script to generate the simulation results.

    ```bash
    python Cash_Flower.py

3. **Analyze the Results:** The simulation will produce visualizations showing monthly cash flows (absolute and statistical aggregated), accumulated funds, and the risk of negative balance.

4. **Save or Modify Plots:** The script generates and saves a PNG file with the plots, which can be easily integrated into reports or presentations.

5. **Don't forget to monitor frequently:** These plots should help you evaluate the financial health of your group, discuss needs and opportunities with your funders (governmental agencies, institute directors, company upper admnistrators, etc.) but the data and plots should be updated in a cadence. Quarterly is a suggestion, but adapt according to your needs!

## Example

Here’s how you might configure data:

```python
singles_params = [
    {'name': 'grant_test', 'chance': 0.8, 'month': 1, 'amount': 1000000},
    {'name': 'big_sale', 'chance': 0.5, 'month': 8, 'amount': 500000},
]

recurrent_params = [
    {'name': 'scholarship', 'chance': 0.7, 'month': 2, 'amount': 1000000, 'duration': 48, 'number': 3, 'payment_frequency': 3},
]

stable_sources = [
    {'name': 'Personnel', 'avg_value': 600000, 'std_value': 100000, 'type': 'expense'},
]

projects = [
    {
        'name': 'Project A',
        'start_chance': 1.0,
        'kill_chance': 0.1,
        'start_month': 1,
        'milestones': [
            {'name': 'milestone_1', 'month': 1, 'amount': 700000, 'success_chance': 0.9},
        ]
    }
]
```
## Contributing

Contributions are welcome! Feel free to open issues, submit pull requests, or suggest new features. Among my list of ideas, I have:

- Improve histogram of monthly expenses and gains.
- Change data gathering and reading so users can take resources from Excel spreadsheets or Pandas dataframes.
- Plot categorized data according to the different types of cashflow events, evaluating how important each is at a given time.
- Please, report any use case that is not being supported by this program. But also bear in mind that users need to adapt to capabilities as well.

