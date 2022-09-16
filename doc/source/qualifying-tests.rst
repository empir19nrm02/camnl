Qualifying Tests
================

Before running the methods for the determination of the non-linearity, some requirements have to be checked.
(all applies to raw camera-/sensor-images)

1. Test for steps in the internal timing. 

   The relative timing of the integration times is required to be sufficiently
   exact because it's the foundation of the calculations. Usually the internal
   timing signal used for the integration time is not acessable. Therefore
   inconsistencies have to be tested in an indirect way.
   `NLPlot.plot_sequence_diff_to_mean()` implements one approach. For each
   integration time series all repetitive images are read in and averaged. The
   average values of all integration times where no overload is reached (in the
   images itself) are used to apply a linear regression. With this regression
   parameters the expected average mean value for each integration time can be
   calculated. The difference of each average value to this expected value is
   plotted, once over integration time and once over counts.

   The curve should be smooth from one integration time to the next. When steps
   occur this can be an indicator for timing inconsistensies, internal range
   switches or compensations.

2. Test for the Gaussion distribution of the dark noise.

   The dark signal should be gaussion distributed. This requires that the
   signal is shifted by an offset so that no clipping occurs at 0-value. This
   means the offset should be larger than the distribution with. Or in other
   words: No pixel should reach the 0-value.
   
   `NLPlot.plot_darksignal_stats()` calculates some statistics on the noise
   and plots images where pixel below a given threshold are marked. 

3. Test for dark signal dependency over time.

   `NLPLot.plot_darksignal_over_inttime()` plots the averaged images for each
   integration time (for short and long series). For short integration time the
   data should be usable to extrapolate to $t_\mathrm{i}=0$.  For long
   integration time the values should increase or stay constant.  A relatively
   large increase may indicate that no internal compensation is done.  A
   constant value means that there is a compensation. On one hand this is good
   but it also means that there are components that are not accessible for
   modelling.  If the dark signal decreases over time it indicates an
   overcompensation.


**API has to be revised! Currently uses hard coded series names.**

