import pandas as pd
import matplotlib.pyplot as plt


def visualize_profit_by_state(states_in_loss):
    states_in_loss.plot.bar(x="State", y="Profit", rot=45, title="States with negative average profit")
    plt.xlabel("States")
    plt.ylabel("Average profit (negative)")
    plt.tight_layout()
    plt.savefig(f'{output_dir}task3_states_profits.png')
    # plt.show()


def analyze_by_states(data):
    mean_profit_by_state = data.groupby(data['State'], as_index=False)["Profit"].mean()
    states_in_loss = mean_profit_by_state.loc[mean_profit_by_state["Profit"] < 0].sort_values(by="Profit")
    visualize_profit_by_state(states_in_loss)


def visualize_profit_by_discount(discounts_profit_relation):
    discounts_profit_relation.plot.bar(x="Discount", y="Profit", rot=45, title="Impact of discount on profit")
    plt.xlabel("Discount")
    plt.ylabel("Average profit (negative)")
    plt.tight_layout()
    plt.savefig(f'{output_dir}task3_discount_profits.png')
    # plt.show()


def analyze_by_discount(data):
    mean_profit_by_disc = data.groupby(data['Discount'], as_index=False)["Profit"].mean()
    discounts_profit_relation = mean_profit_by_disc.loc[mean_profit_by_disc["Profit"] < 0].sort_values(by="Profit")
    visualize_profit_by_discount(discounts_profit_relation)


def visualize_profit_by_sub_category(sub_categories_with_min_profit):
    sub_categories_with_min_profit.plot.bar(x="Sub-Category", y="Profit", rot=0,
                                            title="Sub-Categories with negative average profit")
    plt.xlabel("Sub-Category")
    plt.ylabel("Average profit (negative)")
    plt.tight_layout()
    plt.savefig(f'{output_dir}task3_subcat_profits.png')
    # plt.show()


def analyze_by_sub_category(data):
    mean_profit_by_sub_cat = data.groupby(data['Sub-Category'], as_index=False)["Profit"].mean()
    sub_categories_with_min_profit = mean_profit_by_sub_cat.loc[mean_profit_by_sub_cat["Profit"] < 0].sort_values(by="Profit")
    visualize_profit_by_sub_category(sub_categories_with_min_profit)


def load_data(path):
    data = pd.read_csv(path)
    return data


def exploratory_data_analysis_retail():
    path = "./datasets/SampleSuperstore.csv"
    data = load_data(path)
    analyze_by_states(data)
    analyze_by_discount(data)
    analyze_by_sub_category(data)
    print("Done")


if __name__ == "__main__":
    output_dir = './outputs/'
    exploratory_data_analysis_retail()
