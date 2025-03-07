from skyfield.api import load, Topos, EarthSatellite, utc
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PassDetails:
    start_time: datetime
    end_time: datetime
    duration: float  # minutes
    max_elevation: float
    min_elevation: float
    is_sunlit: bool
    angular_speed: float  # degrees per second
    fov_duration: float  # minutes the target will stay in FOV when tracked
    peak_ra: float  # hours
    peak_dec: float  # degrees
    peak_alt: float  # degrees
    peak_az: float  # degrees
    
class SatelliteVisibilityPredictor:
    def __init__(self, lat: float, lon: float, elevation: float, fov_deg: float,
                 min_elevation_deg: float = 30.0):
        """Initialize the satellite visibility predictor."""
        self.ts = load.timescale()
        self.observer = Topos(
            latitude_degrees=lat,
            longitude_degrees=lon,
            elevation_m=elevation
        )
        self.fov_deg = fov_deg
        self.min_elevation_deg = min_elevation_deg
        self.eph = load('de421.bsp')

    def load_tle_file(self, filename: str) -> List[EarthSatellite]:
        """Load TLE data from a file."""
        satellites = []
        with open(filename, 'r') as f:
            lines = f.readlines()
            
        for i in range(0, len(lines), 3):
            if i + 2 >= len(lines):
                break
                
            name = lines[i].strip()
            line1 = lines[i + 1].strip()
            line2 = lines[i + 2].strip()
            
            try:
                sat = EarthSatellite(line1, line2, name, self.ts)
                satellites.append(sat)
            except Exception as e:
                print(f"Error loading satellite {name}: {e}")
                
        return satellites

    def get_night_interval(self, date: datetime) -> Tuple[datetime, datetime]:
        """Calculate the night observing interval (sunset to sunrise)."""
        from astral import LocationInfo, sun
        
        loc = LocationInfo(
            latitude=self.observer.latitude.degrees,
            longitude=self.observer.longitude.degrees,
            elevation=self.observer.elevation.m
        )
        
        # Get sunset and sunrise times
        s = sun.sun(loc.observer, date=date.date())
        sunset = s['sunset'].replace(tzinfo=utc)
        sunrise = s['sunrise'].replace(tzinfo=utc)
        
        # If the time is after sunset, use next day's sunrise
        if date > sunset:
            next_day = date.date() + timedelta(days=1)
            sunrise = sun.sun(loc.observer, date=next_day)['sunrise'].replace(tzinfo=utc)
            
        return sunset, sunrise

    def calculate_angular_speed(self, positions: List[Tuple], times: List[float]) -> float:
        """Calculate average angular speed of the satellite."""
        if len(positions) < 2:
            return 0.0
            
        speeds = []
        for i in range(len(positions)-1):
            ra1, dec1 = positions[i]
            ra2, dec2 = positions[i+1]
            dt = times[i+1] - times[i]
            
            # Convert RA to same hemisphere if crossing RA=0/24h boundary
            if abs(ra2 - ra1) > 12:
                if ra1 > 12:
                    ra2 += 24
                else:
                    ra1 += 24
            
            # Calculate angular separation
            separation = np.arccos(
                np.sin(np.radians(dec1)) * np.sin(np.radians(dec2)) +
                np.cos(np.radians(dec1)) * np.cos(np.radians(dec2)) *
                np.cos(np.radians((ra2 - ra1) * 15))  # Convert RA hours to degrees
            )
            separation_deg = np.degrees(separation)
            
            if dt > 0:  # Avoid division by zero
                speeds.append(separation_deg / dt)
            
        return np.median(speeds) if speeds else 0.0  # Use median for robustness

    def calculate_fov_duration(self, angular_speed: float) -> float:
        """Calculate time a target will stay in FOV."""
        if angular_speed <= 0:
            return 0.0
        
        # For very slow-moving objects (like GEO sats), cap the duration
        if angular_speed < 0.0001:  # Less than 0.36 degrees per hour
            return None  # Cap at 4 hours
            
        # Calculate time to cross FOV
        # Use 0.8 * FOV to ensure good visibility (not right at the edge)
        fov_crossing_time = (0.8 * self.fov_deg) / angular_speed  # seconds
        return min(fov_crossing_time / 60.0, 240.0)  # Convert to minutes, cap at 4 hours


    def analyze_satellite_pass(self, satellite: EarthSatellite,
                             start_time: datetime, end_time: datetime,
                             time_step: int = 60) -> Optional[PassDetails]:
        """Analyze a single satellite pass in detail."""
        t_start = self.ts.from_datetime(start_time)
        t_end = self.ts.from_datetime(end_time)
        
        # Sample points during the pass
        num_points = int((end_time - start_time).total_seconds() / time_step) + 1
        times = []
        positions = []
        elevations = []
        sunlit_samples = 0
        peak_ra = peak_dec = peak_alt = peak_az = 0
        max_elevation = -90
        min_elevation = 90
        
        for i in range(num_points):
            t = self.ts.from_datetime(start_time + timedelta(seconds=i * time_step))
            times.append(i * time_step)
            
            difference = satellite - self.observer
            topocentric = difference.at(t)
            
            # Get positions
            ra, dec, _ = topocentric.radec()
            alt, az, _ = topocentric.altaz()
            
            positions.append((ra.hours, dec.degrees))
            elevations.append(alt.degrees)
            
            # Track peak position
            if alt.degrees > max_elevation:
                max_elevation = alt.degrees
                peak_ra = ra.hours
                peak_dec = dec.degrees
                peak_alt = alt.degrees
                peak_az = az.degrees
                
            min_elevation = min(min_elevation, alt.degrees)
            
            # Check if sunlit
            if satellite.at(t).is_sunlit(self.eph):
                sunlit_samples += 1
                
        # Calculate metrics
        duration = (end_time - start_time).total_seconds() / 60.0  # minutes
        angular_speed = self.calculate_angular_speed(positions, times)
        fov_duration = self.calculate_fov_duration(angular_speed)
        is_sunlit = (sunlit_samples / num_points) > 0.5
        
        return PassDetails(
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            max_elevation=max_elevation,
            min_elevation=min_elevation,
            is_sunlit=is_sunlit,
            angular_speed=angular_speed,
            fov_duration=fov_duration,
            peak_ra=peak_ra,
            peak_dec=peak_dec,
            peak_alt=peak_alt,
            peak_az=peak_az
        )

    def find_continuous_visibility_periods(self, satellite: EarthSatellite,
                                        t0: datetime, t1: datetime,
                                        time_step: int = 300) -> List[PassDetails]:
        """Find periods of continuous visibility including multiple passes."""
        periods = []
        is_visible = False
        start_time = None
        
        current_time = t0
        while current_time <= t1:
            t = self.ts.from_datetime(current_time)
            difference = satellite - self.observer
            topocentric = difference.at(t)
            alt, _, _ = topocentric.altaz()
            
            # Check if satellite is above minimum elevation
            if alt.degrees >= self.min_elevation_deg:
                if not is_visible:
                    start_time = current_time
                    is_visible = True
            elif is_visible:
                # End of visible period
                periods.append(self.analyze_satellite_pass(
                    satellite, start_time, current_time
                ))
                is_visible = False
                
            current_time += timedelta(seconds=time_step)
            
        # Handle case where satellite is still visible at end of time window
        if is_visible and start_time is not None:
            periods.append(self.analyze_satellite_pass(
                satellite, start_time, t1
            ))
            
        return periods

    def compute_visibility_score(self, satellite: EarthSatellite,
                               current_time: datetime,
                               periods: List[PassDetails]) -> Dict:
        """Compute visibility metrics for a satellite."""
        if not periods:
            return {
                'total_fov_time': 0,
                'next_pass': None,
                'all_passes': [],
                'score': 0
            }
        
        # Calculate total potential FOV time
        total_fov_time = sum(p.fov_duration for p in periods)
        
        # Find next pass
        future_periods = [p for p in periods if p.start_time > current_time]
        next_pass = min(future_periods, key=lambda p: p.start_time) if future_periods else None
        
        # Calculate score based on:
        # 1. Total potential observation time
        # 2. Time until next pass
        # 3. Maximum elevation achieved
        # 4. Angular speed (prefer slower moving objects)
        score = total_fov_time
        if next_pass:
            time_to_next = (next_pass.start_time - current_time).total_seconds() / 3600
            if time_to_next <= 48:
                score *= (1 + (48 - time_to_next) / 48)
                
            # Bonus for higher elevation
            score *= (1 + (next_pass.max_elevation - self.min_elevation_deg) / 90)
            
            # Bonus for slower moving objects
            if next_pass.angular_speed > 0:
                score *= (1 + 1 / (next_pass.angular_speed * 3600))  # Convert to deg/hour
                
        return {
            'total_fov_time': total_fov_time,
            'next_pass': next_pass,
            'all_passes': periods,
            'score': score
        }

    def find_best_satellites(self, satellites: List[EarthSatellite],
                           current_time: datetime,
                           duration_hours: float = 48,
                           max_candidates: int = 10) -> List[Dict]:
        """Find the best satellites for observation."""
        t0 = current_time
        t1 = t0 + timedelta(hours=duration_hours)
        
        candidates = []
        for sat in satellites:
            # Find all visibility periods
            periods = self.find_continuous_visibility_periods(sat, t0, t1)
            
            # Compute visibility score
            visibility = self.compute_visibility_score(sat, current_time, periods)
            
            if visibility['score'] > 0:
                candidates.append({
                    'satellite': sat,
                    'visibility': visibility
                })
        
        # Sort by score in descending order
        candidates.sort(key=lambda x: x['visibility']['score'], reverse=True)
        return candidates[:max_candidates]