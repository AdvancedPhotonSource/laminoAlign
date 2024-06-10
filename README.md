# laminoAlign

laminoAlign is a python software package for aligning projections from laminography experiments.

Portions of this software package are based on the cSAXS ptychography MATLAB package developed by the Science IT and the coherent X-ray scattering (CXS) groups at the Paul Scherrer Institut in Switzerland.[https://www.psi.ch/en/sls/csaxs/software](https://www.psi.ch/en/sls/csaxs/software). 

For more details on the alignment methods used in this software package, see this paper:
[Michal Odstrčil, Mirko Holler, Jörg Raabe, and Manuel Guizar-Sicairos, "Alignment methods for nanotomography with deep subpixel accuracy," Opt. Express **27**, 36637-36652 (2019)](https://opg.optica.org/oe/fulltext.cfm?uri=oe-27-25-36637&id=423764)
## Standard Installation

1. Install [miniforge](https://github.com/conda-forge/miniforge)
2. Clone the directory
```
$ git clone https://github.com/AdvancedPhotonSource/laminoAlign.git
```
4. Create a new environment and install the requirements that need to be installed from the conda-forge channel
```
 $ conda create -c conda-forge -n laminoAlign --file requirements.txt
```
3. Install laminoAlign
```
$ pip install .
```

For developer installation, use `pip install -e .`

## Acknowledgements
Portions of laminoAlign are based on the cSAXS MATLAB package developed by the CXS group at the Paul Scherrer Institut. For this reason, a redistribution notice and the full PtychoShelves license.

**Redistribution Notice:**
```
*-----------------------------------------------------------------------*
|                                                                       |
|  Except where otherwise noted, this work is licensed under a          |
|  Creative Commons Attribution-NonCommercial-ShareAlike 4.0            |
|  International (CC BY-NC-SA 4.0) license.                             |
|                                                                       |
|  Copyright (c) 2017 by Paul Scherrer Institute (http://www.psi.ch)    |
|                                                                       |
|       Author: CXS group, PSI                                          |
*-----------------------------------------------------------------------*
You may use this code with the following provisions:

If the code is fully or partially redistributed, or rewritten in another
  computing language this notice should be included in the redistribution.

If this code, or subfunctions or parts of it, is used for research in a
  publication or if it is fully or partially rewritten for another
  computing language the authors and institution should be acknowledged
  in written form in the publication: “Data processing was carried out
  using the “cSAXS matlab package” developed by the CXS group,
  Paul Scherrer Institut, Switzerland.”
  Variations on the latter text can be incorporated upon discussion with
  the CXS group if needed to more specifically reflect the use of the package
  for the published work.

A publication that focuses on describing features, or parameters, that
   are already existing in the code should be first discussed with the
   authors.

This code and subroutines are part of a continuous development, they
   are provided “as they are” without guarantees or liability on part
   of PSI or the authors. It is the user responsibility to ensure its
   proper use and the correctness of the results.
```

**PtychoShelves License Agreement**
Introduction 

This license agreement sets forth the terms and conditions under which the PAUL SCHERRER INSTITUT (PSI), CH-5232 Villigen-PSI, Switzerland (hereafter "LICENSOR") will grant you (hereafter "LICENSEE") a royalty-free, non-exclusive license for academic, non-commercial purposes only (hereafter "LICENSE") to use the cSAXS ptychography MATLAB package computer software program and associated documentation furnished hereunder (hereafter "PROGRAM").

Terms and Conditions of the LICENSE
1.	LICENSOR grants to LICENSEE a royalty-free, non-exclusive license to use the PROGRAM for academic, non-commercial purposes, upon the terms and conditions hereinafter set out and until termination of this license as set forth below.
2.	LICENSEE acknowledges that the PROGRAM is a research tool still in the development stage. The PROGRAM is provided without any related services, improvements or warranties from LICENSOR and that the LICENSE is entered into in order to enable others to utilize the PROGRAM in their academic activities. It is the LICENSEE's responsibility to ensure its proper use and the correctness of the results.
3.	THE PROGRAM IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT OF ANY PATENTS, COPYRIGHTS, TRADEMARKS OR OTHER RIGHTS. IN NO EVENT SHALL THE LICENSOR, THE AUTHORS OR THE COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DIRECT, INDIRECT OR CONSEQUENTIAL DAMAGES OR OTHER LIABILITY ARISING FROM, OUT OF OR IN CONNECTION WITH THE PROGRAM OR THE USE OF THE PROGRAM OR OTHER DEALINGS IN THE PROGRAM.
4.	LICENSEE agrees that it will use the PROGRAM and any modifications, improvements, or derivatives of PROGRAM that LICENSEE may create (collectively, "IMPROVEMENTS") solely for academic, non-commercial purposes and that any copy of PROGRAM or derivatives thereof shall be distributed only under the same license as PROGRAM. The terms "academic, non-commercial", as used in this Agreement, mean academic or other scholarly research which (a) is not undertaken for profit, or (b) is not intended to produce works, services, or data for commercial use, or (c) is neither conducted, nor funded, by a person or an entity engaged in the commercial use, application or exploitation of works similar to the PROGRAM.
5.	LICENSEE agrees that it shall make the following acknowledgement in any publication resulting from the use of the PROGRAM or any translation of the code into another computing language:
"Data processing was carried out using the cSAXS ptychography MATLAB package developed by the Science IT and the coherent X-ray scattering (CXS) groups, Paul Scherrer Institut, Switzerland."

Additionally, any publication using the package, or any translation of the code into another computing language should cite

(for PtychoShelves) K. Wakonig, H.-C. Stadler, M. Odstrčil, E.H.R. Tsai, A. Diaz, M. Holler, I. Usov, J. Raabe, A. Menzel, M. Guizar-Sicairos, PtychoShelves, a versatile high-level framework for high-performance analysis of ptychographic data, J. Appl. Cryst. 53(2) (2020). (doi: 10.1107/S1600576720001776)

(for difference map) P. Thibault, M. Dierolf, A. Menzel, O. Bunk, C. David, F. Pfeiffer, High-resolution scanning X-ray diffraction microscopy, Science 321, 379-382 (2008). (doi: 10.1126/science.1158573).

(for maximum likelihood) P. Thibault and M. Guizar-Sicairos, Maximum-likelihood refinement for coherent diffractive imaging, New J. Phys. 14, 063004 (2012). (doi: 10.1088/1367-2630/14/6/063004).

(for mixed coherent modes) P. Thibault and A. Menzel, Reconstructing state mixtures from diffraction measurements, Nature 494, 68-71 (2013). (doi: 10.1038/nature11806).

(and/or for multislice) E. H. R. Tsai, I. Usov, A. Diaz, A. Menzel, and M. Guizar-Sicairos, X-ray ptychography with extended depth of field, Opt. Express 24, 29089-29108 (2016). (doi: 10.1364/OE.24.029089).

6.	Except for the above-mentioned acknowledgment, LICENSEE shall not use the PROGRAM title or the names or logos of LICENSOR, nor any adaptation thereof, nor the names of any of its employees or laboratories, in any advertising, promotional or sales material without prior written consent obtained from LICENSOR in each case.
7.	Ownership of all rights, including copyright in the PROGRAM and in any material associated therewith, shall at all times remain with LICENSOR, and LICENSEE agrees to preserve same. LICENSEE agrees not to use any portion of the PROGRAM or of any IMPROVEMENTS in any machine-readable form outside the PROGRAM, nor to make any copies except for its internal use, without prior written consent of LICENSOR. LICENSEE agrees to place the following copyright notice on any such copies: 
@ All rights reserved. PAUL SCHERRER INSTITUT, Switzerland, Laboratory for Macromolecules and Bioimaging, 2017. 
8.	The LICENSE shall not be construed to confer any rights upon LICENSEE by implication or otherwise except as specifically set forth herein.
9.	DISCLAIMER: LICENSEE shall be aware that Phase Focus Limited of Sheffield, UK has an international portfolio of patents and pending applications which relate to ptychography and that the PROGRAM may be capable of being used in circumstances which may fall within the claims of one or more of the Phase Focus patents, in particular of patent with international application number PCT/GB2005/001464. The LICENSOR explicitly declares not to indemnify the users of the software in case Phase Focus or any other third party will open a legal action against the LICENSEE due to the use of the program.
10.	This Agreement shall be governed by the material laws of Switzerland and any dispute arising out of this Agreement or use of the PROGRAM shall be brought before the courts of Zurich, Switzerland.
