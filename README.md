# MicroMet

![Read the Docs](https://img.shields.io/readthedocs/micromet)
![PyPI - Version](https://img.shields.io/pypi/v/micromet)
<table><tr><td>All platforms:</td>
    <td>
      <a href="https://dev.azure.com/conda-forge/feedstock-builds/_build/latest?definitionId=25590&branchName=main">
        <img src="https://dev.azure.com/conda-forge/feedstock-builds/_apis/build/status/micromet-feedstock?branchName=main">
      </a>
    </td>
  </tr>
</table>

Current Conda release info
====================

| Name | Downloads | Version | Platforms |
| --- | --- | --- | --- |
| [![Conda Recipe](https://img.shields.io/badge/recipe-micromet-green.svg)](https://anaconda.org/conda-forge/micromet) | [![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/micromet.svg)](https://anaconda.org/conda-forge/micromet) | [![Conda Version](https://img.shields.io/conda/vn/conda-forge/micromet.svg)](https://anaconda.org/conda-forge/micromet) | [![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/micromet.svg)](https://anaconda.org/conda-forge/micromet) |
# Description

Scripts to process half-hourly Eddy Covariance data from [Campbell Scientific CR6 dataloggers](https://www.campbellsci.com/cr6) running [EasyFluxDL](https://www.campbellsci.com/easyflux-dl) for submission to the Ameriflux Data Portal.

Utah Flux Network stations have a dual-CR6 setup, where one CR6 runs the EasyFluxDL program, and the other CR6 collects redundant meteorological data.

These scripts can add missing headers and compile existing downloads of half-hourly data.  They help the data conform to Ameriflux's formatting standards.

# Documentation
Documentation can be found on [readthedocs](https://micromet.readthedocs.io/en/latest/)

# Data Processing Workflow
1. Process data on the fly using EasyfluxDL; Provide immediate data through UGS portal 
2. QA/QC processed data to see if it meets quality checks
3. Reprocess data manually, focusing on low quality datasets
4. Upload refined data to Ameriflux

# Ameriflux
* [Levels of data processing](https://ameriflux.lbl.gov/data/aboutdata/data-processing-levels/)
* [Ameriflux data pipeline](https://ameriflux.lbl.gov/data/data-processing-pipelines/)

