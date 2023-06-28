# CO2 Dashboard

Webpage Link: https://net0thon.streamlit.app/

Description:
This repo is a web app of a dashboard that uses ML to predict the population and CO2 emission rates in Saudi Arabia. 
It also tracks real CO2 abatement data from local efforts, and has gives you the ability to build CO2 abatement scenarios
using geothermal wells and CO2 sequestration wells. This webpage was built as part of a solution to a climate-change hackathon (*Net0thon*) organized by Saudi Aramco in Dhahran, Saudi Arabia on May 28th, 2023.

Objective:
Using this webpage you can test different scenarios of CO2 abatement strategies to accelarate the Kingdom's targets of:
1. Reducing/offsetting emissions by 278 million tons of CO2 by 2030
2. Acheive net-zero by 2060

Features:
- Graph of historical population and CO2 emissions (source: [World Bank](https://data.worldbank.org/))
- AI/ML to predict future forecasts of population, CO2 emissions and CO2 abatement rates in Saudi Arabia using facebook [prophet](https://facebook.github.io/prophet/) 
- Map visualization of CO2 emissions from the leading industrial sectors (source: [Rowaihy et al., 2022](https://www.sciencedirect.com/science/article/pii/S2590174522001222))
- Collected data from local CO2 sequesteration and utilization efforts, including: Saudi Aramco, SABIC, and Ministry of Agriculture
