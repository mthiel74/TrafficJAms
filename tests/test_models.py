"""Unit tests for all traffic models."""

import numpy as np
import pytest
import os
import tempfile


class TestLWR:
    def test_simulate_returns_required_keys(self):
        from trafficjams import lwr
        r = lwr.simulate(nx=50, T=0.1)
        assert "x" in r and "t" in r and "density" in r

    def test_density_within_bounds(self):
        from trafficjams import lwr
        r = lwr.simulate(nx=50, T=0.1, rho_max=150)
        assert r["density"].min() >= 0
        assert r["density"].max() <= 150

    def test_greenshields_zero_density(self):
        from trafficjams.lwr import greenshields
        assert greenshields(0, 150, 30) == 0

    def test_greenshields_max_density(self):
        from trafficjams.lwr import greenshields
        assert greenshields(150, 150, 30) == 0

    def test_plot_creates_file(self):
        from trafficjams import lwr
        r = lwr.simulate(nx=50, T=0.05)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            lwr.plot(r, save_path=f.name)
            assert os.path.getsize(f.name) > 0
            os.unlink(f.name)


class TestPayneWhitham:
    def test_simulate_returns_density_and_velocity(self):
        from trafficjams import payne_whitham
        r = payne_whitham.simulate(nx=50, T=0.05)
        assert "density" in r and "velocity" in r

    def test_density_positive(self):
        from trafficjams import payne_whitham
        r = payne_whitham.simulate(nx=50, T=0.05)
        assert r["density"].min() >= 0

    def test_velocity_non_negative(self):
        from trafficjams import payne_whitham
        r = payne_whitham.simulate(nx=50, T=0.05)
        assert r["velocity"].min() >= 0


class TestIDM:
    def test_simulate_returns_trajectories(self):
        from trafficjams import idm
        r = idm.simulate(n_vehicles=10, T=10)
        assert r["positions"].shape[1] == 10
        assert r["velocities"].shape[1] == 10

    def test_velocities_non_negative(self):
        from trafficjams import idm
        r = idm.simulate(n_vehicles=10, T=20)
        assert r["velocities"].min() >= 0

    def test_positions_within_road(self):
        from trafficjams import idm
        r = idm.simulate(n_vehicles=10, road_length=500, T=10)
        assert r["positions"].max() < 500
        assert r["positions"].min() >= 0


class TestBando:
    def test_optimal_velocity_monotone(self):
        from trafficjams.bando import optimal_velocity
        s = np.linspace(1, 100, 50)
        v = optimal_velocity(s)
        assert np.all(np.diff(v) >= 0)  # monotonically increasing

    def test_simulate_basic(self):
        from trafficjams import bando
        r = bando.simulate(n_vehicles=10, T=20)
        assert r["velocities"].min() >= 0


class TestNagelSchreckenberg:
    def test_simulate_spacetime_shape(self):
        from trafficjams import nagel_schreckenberg
        r = nagel_schreckenberg.simulate(road_length=100, n_vehicles=20, T=50)
        assert r["spacetime"].shape == (50, 100)

    def test_speed_within_bounds(self):
        from trafficjams import nagel_schreckenberg
        r = nagel_schreckenberg.simulate(road_length=100, n_vehicles=20, v_max=5, T=50)
        occupied = r["spacetime"] >= 0
        assert r["spacetime"][occupied].max() <= 5
        assert r["spacetime"][occupied].min() >= 0

    def test_vehicle_count_conserved(self):
        from trafficjams import nagel_schreckenberg
        r = nagel_schreckenberg.simulate(road_length=200, n_vehicles=30, T=100)
        for t in range(r["T"]):
            n_occupied = (r["spacetime"][t] >= 0).sum()
            assert n_occupied == 30


class TestNetworkAssignment:
    def test_flow_conservation(self):
        from trafficjams import network_assignment
        r = network_assignment.simulate()
        total_path_flow = r["path_flows"].sum()
        assert abs(total_path_flow - r["total_demand"]) < 1.0

    def test_wardrop_equilibrium(self):
        from trafficjams import network_assignment
        r = network_assignment.simulate()
        used = r["path_flows"] > 10  # paths with significant flow
        if used.sum() > 1:
            costs = r["path_costs"][used]
            assert costs.max() - costs.min() < 2.0  # approximately equal costs

    def test_flows_non_negative(self):
        from trafficjams import network_assignment
        r = network_assignment.simulate()
        assert np.all(r["link_flows"] >= -1e-6)


class TestQueueing:
    def test_queue_length_increases_with_rho(self):
        from trafficjams.queueing import md1_queue_length
        rhos = [0.2, 0.5, 0.8]
        ql = [md1_queue_length(r) for r in rhos]
        assert ql[0] < ql[1] < ql[2]

    def test_simulate_returns_expected_keys(self):
        from trafficjams import queueing
        r = queueing.simulate()
        assert "rho" in r and "queue_length" in r and "waiting_time" in r

    def test_queue_diverges_near_capacity(self):
        from trafficjams.queueing import md1_queue_length
        assert md1_queue_length(0.99) > 10


class TestNaSch2Lane:
    def test_simulate_runs(self):
        from trafficjams import nagel_schreckenberg_2lane
        r = nagel_schreckenberg_2lane.simulate(road_length=100, n_vehicles=30, T=50)
        assert r["spacetime"].shape == (50, 2, 100)


class TestDynamicAssignment:
    def test_simulate_returns_time_series(self):
        from trafficjams import dynamic_assignment
        r = dynamic_assignment.simulate(n_periods=5)
        assert r["link_flows"].shape[0] == 5
        assert r["path_flows"].shape[0] == 5
