import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
from satellite_visibility import SatelliteVisibilityPredictor
from datetime import datetime, timedelta
from skyfield.api import utc, load
import numpy as np
from skyfield.almanac import dark_twilight_day
import astropy.units as u
import os

class SatelliteVisibilityGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Satellite Visibility Predictor")
       
        # Set smaller window size
        self.root.minsize(1200, 600)  # Reduced from 800 to 600
       
        # Configure grid weight for responsiveness
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(2, weight=1)
       
        # Current time and user
        self.current_time = datetime.strptime("2025-03-07 15:55:19", "%Y-%m-%d %H:%M:%S").replace(tzinfo=utc)
        self.current_user = "bikramATastro"
       
        # Initialize other attributes
        self.predictor = SatelliteVisibilityPredictor(
            lat=29.361568488068865,
            lon=79.685018631768124,
            elevation=2400.0,
            fov_deg=2.0,
            min_elevation_deg=30.0
        )
       
        self.all_candidates = []
        self.create_gui_elements()

    def configure_styles(self):
        """Configure custom styles for better appearance"""
        self.style.configure('TLabel', font=('Helvetica', 10))
        self.style.configure('TButton', font=('Helvetica', 10))
        self.style.configure('Header.TLabel', font=('Helvetica', 12, 'bold'))
        self.style.configure('Title.TLabel', font=('Helvetica', 14, 'bold'))
       
        # Configure Treeview colors
        self.style.configure('Treeview',
                           background='#f0f0f0',
                           fieldbackground='#ffffff',
                           font=('Helvetica', 10))
        self.style.configure('Treeview.Heading',
                           font=('Helvetica', 10, 'bold'))
       
        # Configure special row colors for night passes
        self.style.map('Treeview',
                      background=[('selected', '#0078D7')],
                      foreground=[('selected', 'white')])
 
       
    def create_gui_elements(self):
        # Main container frame with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        main_frame.grid_columnconfigure(0, weight=1)
       
        # Title Frame with improved styling
        title_frame = ttk.Frame(main_frame)
        title_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
       
        ttk.Label(title_frame,
                 text="Satellite Visibility Predictor - ARIES",
                 style='Title.TLabel').grid(row=0, column=0, pady=(0, 5))
       
        # Info Frame with current date and user
        info_frame = ttk.Frame(main_frame)
        info_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        info_frame.grid_columnconfigure(1, weight=1)
       
        ttk.Label(info_frame,
                text=f"Current Date and Time (UTC):",
                style='Header.TLabel').grid(row=0, column=0, padx=(0, 5))
        ttk.Label(info_frame,
                text=self.current_time.strftime('%Y-%m-%d %H:%M:%S'),
                style='TLabel').grid(row=0, column=1, sticky="w")
       
        ttk.Label(info_frame,
                text="Current User:",
                style='Header.TLabel').grid(row=1, column=0, padx=(0, 5))
        ttk.Label(info_frame,
                text=self.current_user,
                style='TLabel').grid(row=1, column=1, sticky="w")
       
        # Settings Frame with improved layout
        settings_frame = ttk.LabelFrame(main_frame, text="Display Settings", padding="10")
        settings_frame.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        settings_frame.grid_columnconfigure(1, weight=1)
       
        # Unit selection
        ttk.Label(settings_frame, text="Angle Units:").grid(row=0, column=0, padx=5)
        self.angle_unit_var = tk.StringVar(value="deg")
        ttk.Radiobutton(settings_frame, text="Degrees",
                       variable=self.angle_unit_var, value="deg",
                       command=self.update_table).grid(row=0, column=1)
        ttk.Radiobutton(settings_frame, text="Radians",
                       variable=self.angle_unit_var, value="rad",
                       command=self.update_table).grid(row=0, column=2)
       
        ttk.Label(settings_frame, text="Speed Units:").grid(row=1, column=0, padx=5)
        self.speed_unit_var = tk.StringVar(value="arcmin/s")
        ttk.Radiobutton(settings_frame, text="Arcmin/s",
                       variable=self.speed_unit_var, value="arcmin/s",
                       command=self.update_table).grid(row=1, column=1)
        ttk.Radiobutton(settings_frame, text="Deg/hour",
                       variable=self.speed_unit_var, value="deg/h",
                       command=self.update_table).grid(row=1, column=2)
       
        # Sun elevation setting
        ttk.Label(settings_frame, text="Min Sun Elevation (deg):").grid(row=2, column=0, padx=5)
        self.sun_elevation_var = tk.StringVar(value="-18")
        sun_elevation_entry = ttk.Entry(settings_frame, textvariable=self.sun_elevation_var, width=10)
        sun_elevation_entry.grid(row=2, column=1)
       
        # Filter settings
        filter_frame = ttk.LabelFrame(settings_frame, text="Filter Settings", padding="5")
        filter_frame.grid(row=3, column=0, columnspan=3, sticky="ew", pady=5)
       
        ttk.Label(filter_frame, text="Max Candidates:").grid(row=0, column=0, padx=5)
        self.max_candidates_var = tk.StringVar(value="50")
        ttk.Entry(filter_frame, textvariable=self.max_candidates_var, width=10).grid(row=0, column=1)
        ttk.Button(filter_frame, text="Apply Filter",
                  command=self.apply_filters).grid(row=0, column=2, padx=5)
       
        # Add night pass highlight toggle
        self.highlight_night_passes = tk.BooleanVar(value=True)
        ttk.Checkbutton(settings_frame,
                       text="Highlight Night Passes",
                       variable=self.highlight_night_passes,
                       command=self.update_table).grid(row=3, column=0, columnspan=2)
       
        # Create resizable table
        self.create_table()
       
        # Bottom frame with action buttons
        bottom_frame = ttk.Frame(main_frame, padding="5")
        bottom_frame.grid(row=4, column=0, sticky="ew", pady=(10, 0))
        bottom_frame.grid_columnconfigure((0,1,2), weight=1)
       
        ttk.Button(bottom_frame, text="Load TLE File",
                  command=self.load_tle).grid(row=0, column=0, padx=5)
        ttk.Button(bottom_frame, text="Generate Look Angles",
                  command=self.generate_look_angles).grid(row=0, column=1, padx=5)
        ttk.Button(bottom_frame, text="Save to CSV",
                  command=self.save_to_csv).grid(row=0, column=2, padx=5)
       
        # Sampling parameters
        sampling_frame = ttk.LabelFrame(self.root, text="Sampling Parameters", padding="5")
        sampling_frame.grid(row=4, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
       
        ttk.Label(sampling_frame, text="Time Step (seconds):").grid(row=0, column=0, padx=5)
        self.time_step_var = tk.StringVar(value="60")
        ttk.Entry(sampling_frame, textvariable=self.time_step_var, width=10).grid(row=0, column=1)

    def create_table(self):
        # Updated columns to include night pass information
        columns = (
            "Satellite", "NORAD ID", "Semi-major Axis (km)",
            "Number of Passes", "Night Passes", "Sunlit Night Passes",
            "Total Duration (min)", "Night Duration (min)",
            "Max El", "Angular Speed", "FOV Duration (min)"
        )
       
        # Create frame for table with scrollbars
        table_frame = ttk.Frame(self.root)
        table_frame.grid(row=3, column=0, sticky="nsew", padx=10, pady=5)
        table_frame.grid_columnconfigure(0, weight=1)
        table_frame.grid_rowconfigure(0, weight=1)
       
        # Create Treeview with scrollbars
        self.tree = ttk.Treeview(table_frame, columns=columns, show='headings',
                               selectmode='extended', height=15)
       
        # Configure scrollbars
        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(table_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
       
        # Grid layout for table and scrollbars
        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
       
        # Configure columns
        for col in columns:
            self.tree.heading(col, text=col, command=lambda c=col: self.sort_treeview(c))
            width = 150 if col in ["Satellite", "Angular Speed"] else 100
            self.tree.column(col, width=width)

    def convert_to_sexagesimal(self, value, is_ra=False):
        """Convert decimal degrees/hours to sexagesimal format."""
        if is_ra:
            # Convert RA from hours to HH:MM:SS
            hours = int(value)
            minutes = int((value - hours) * 60)
            seconds = ((value - hours) * 60 - minutes) * 60
            return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
        else:
            # Convert Dec from degrees to DD:MM:SS
            sign = '-' if value < 0 else '+'
            value = abs(value)
            degrees = int(value)
            minutes = int((value - degrees) * 60)
            seconds = ((value - degrees) * 60 - minutes) * 60
            return f"{sign}{degrees:02d}:{minutes:02d}:{seconds:06.3f}"

    def convert_angle(self, angle_deg):
        """Convert angle based on selected unit."""
        return np.radians(angle_deg) if self.angle_unit_var.get() == "rad" else angle_deg

    def convert_speed(self, speed_deg_per_sec):
        """Convert speed based on selected unit."""
        if self.speed_unit_var.get() == "arcmin/s":
            return speed_deg_per_sec * 60  # Convert to arcmin/s
        return speed_deg_per_sec * 3600  # Convert to deg/hour

    def is_night(self, time):
        """Check if it's astronomical night (sun below specified elevation)."""
        try:
            t = self.predictor.ts.from_datetime(time)
            eph = load('de421.bsp')
            sun = eph['sun']
            earth = eph['earth']
            topos = self.predictor.observer
           
            sun_alt = (earth + topos).at(t).observe(sun).apparent().altaz()[0].degrees
            min_sun_elevation = float(self.sun_elevation_var.get())
            return sun_alt <= min_sun_elevation
        except Exception as e:
            print(f"Error checking night condition: {str(e)}")
            return False

    def update_table(self):
        """Update table with enhanced night pass information"""
        for item in self.tree.get_children():
            self.tree.delete(item)
       
        try:
            max_candidates = int(self.max_candidates_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid max candidates value")
            return
       
        sorted_candidates = sorted(self.all_candidates,
                               key=lambda x: x['visibility']['score'],
                               reverse=True)[:max_candidates]
       
        for candidate in sorted_candidates:
            sat = candidate['satellite']
            visibility = candidate['visibility']
            passes = visibility['all_passes']
           
            if not passes:
                continue
           
            # Calculate night pass statistics
            night_passes = []
            sunlit_night_passes = []
            total_night_duration = 0
           
            for p in passes:
                current_time = p.start_time
                is_night_pass = False
                is_sunlit_night_pass = False
                night_duration = 0
               
                # Check each minute of the pass
                while current_time <= p.end_time:
                    time_step = timedelta(minutes=1)
                    if self.is_night(current_time):
                        is_night_pass = True
                        t = self.predictor.ts.from_datetime(current_time)
                        if sat.at(t).is_sunlit(self.predictor.eph):
                            is_sunlit_night_pass = True
                        night_duration += time_step.total_seconds() / 60
                    current_time += time_step
               
                if is_night_pass:
                    night_passes.append(p)
                    total_night_duration += night_duration
                    if is_sunlit_night_pass:
                        sunlit_night_passes.append(p)
           
            # Prepare display values
            values = (
                sat.name,
                sat.model.satnum,
                f"{sat.model.a * 6378.137:.1f}",
                len(passes),
                len(night_passes),
                len(sunlit_night_passes),
                f"{sum(p.duration for p in passes):.1f}",
                f"{total_night_duration:.1f}",
                f"{self.convert_angle(max(p.max_elevation for p in passes)):.1f}",
                f"{self.convert_speed(max(p.angular_speed for p in passes)):.3f}",
                f"{sum(p.fov_duration if p.fov_duration is not None else 0 for p in passes):.1f}"
            )
           
            # Add item to tree
            item = self.tree.insert('', 'end', values=values)
           
            # Highlight night passes if enabled
            if self.highlight_night_passes.get() and len(sunlit_night_passes) > 0:
                self.tree.tag_configure('night_pass', background='#E6F3FF')
                self.tree.item(item, tags=('night_pass',))

   
    def generate_look_angles(self):
        selected_items = self.tree.selection()
        if not selected_items:
            messagebox.showwarning("Warning", "Please select at least one satellite")
            return
           
        try:
            time_step = int(self.time_step_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid time step value")
            return
           
        try:
            base_dir = filedialog.askdirectory(title="Select base directory for look angles")
            if not base_dir:
                return
               
            # Create options dialog
            dialog = tk.Toplevel(self.root)
            dialog.title("Look Angles Options")
            dialog.transient(self.root)
            dialog.grab_set()
           
            # Night passes option
            night_only_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(
                dialog,
                text="Generate for night passes only",
                variable=night_only_var
            ).pack(pady=5)
           
            # Sunlit only option
            sunlit_only_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(
                dialog,
                text="Generate only when satellite is sunlit",
                variable=sunlit_only_var
            ).pack(pady=5)
           
            # Coordinate format option
            coord_format_frame = ttk.LabelFrame(dialog, text="Coordinate Format", padding="5")
            coord_format_frame.pack(pady=5, padx=5, fill="x")
           
            coord_format_var = tk.StringVar(value="decimal")
            ttk.Radiobutton(
                coord_format_frame,
                text="Decimal (degrees/hours)",
                variable=coord_format_var,
                value="decimal"
            ).pack(anchor="w")
            ttk.Radiobutton(
                coord_format_frame,
                text="Sexagesimal (HH:MM:SS/DD:MM:SS)",
                variable=coord_format_var,
                value="sexagesimal"
            ).pack(anchor="w")
           
            def process_look_angles():
                use_sexagesimal = coord_format_var.get() == "sexagesimal"
                dialog.destroy()
               
                for item in selected_items:
                    values = self.tree.item(item)['values']
                    sat_name = values[0]
                   
                    # Create satellite-specific directory
                    sat_dir = os.path.join(base_dir, sat_name)
                    os.makedirs(sat_dir, exist_ok=True)
                   
                    # Find corresponding candidate
                    candidate = next((c for c in self.all_candidates
                                   if c['satellite'].name == sat_name), None)
                    if not candidate:
                        continue
                   
                    # Filter passes based on conditions
                    passes = candidate['visibility']['all_passes']
                    if night_only_var.get():
                        night_passes = []
                        for p in passes:
                            current_time = p.start_time
                            while current_time <= p.end_time:
                                if self.is_night(current_time):
                                    night_passes.append(p)
                                    break
                                current_time += timedelta(minutes=1)
                        passes = night_passes
                   
                    for pass_detail in passes:
                        look_angles = []
                        current_time = pass_detail.start_time
                       
                        while current_time <= pass_detail.end_time:
                            t = self.predictor.ts.from_datetime(current_time)
                            difference = candidate['satellite'] - self.predictor.observer
                            topocentric = difference.at(t)
                           
                            alt, az, _ = topocentric.altaz()
                            ra, dec, _ = topocentric.radec()
                           
                            # Check if satellite is sunlit
                            is_sunlit = candidate['satellite'].at(t).is_sunlit(self.predictor.eph)
                           
                            # Only include points when conditions are met
                            if (alt.degrees >= self.predictor.min_elevation_deg and
                                (not sunlit_only_var.get() or is_sunlit)):
                               
                                entry = {
                                    'UTC': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                                    'Az_deg': f"{az.degrees:.6f}",
                                    'El_deg': f"{alt.degrees:.6f}",
                                    'Sunlit': 'Yes' if is_sunlit else 'No'
                                }
                               
                                # Convert coordinates if needed
                                if use_sexagesimal:
                                    entry['RA_J2000'] = self.convert_to_sexagesimal(ra.hours, is_ra=True)
                                    entry['Dec_J2000'] = self.convert_to_sexagesimal(dec.degrees, is_ra=False)
                                else:
                                    entry['RA_J2000'] = f"{ra.hours:.6f}"
                                    entry['Dec_J2000'] = f"{dec.degrees:.6f}"
                               
                                look_angles.append(entry)
                           
                            current_time += timedelta(seconds=time_step)
                       
                        # Only write file if we have look angles
                        if look_angles:
                            pass_start = pass_detail.start_time.strftime('%Y%m%d_%H%M%S')
                            filename = os.path.join(sat_dir, f"pass_{pass_start}.csv")
                            df = pd.DataFrame(look_angles)
                           
                            with open(filename, 'w') as f:
                                f.write(f"# Satellite: {sat_name}\n")
                                f.write(f"# Pass Start (UTC): {pass_detail.start_time}\n")
                                f.write(f"# Pass End (UTC): {pass_detail.end_time}\n")
                                f.write(f"# Minimum Elevation: {self.predictor.min_elevation_deg} deg\n")
                                f.write(f"# Time Step: {time_step} seconds\n")
                                f.write(f"# Coordinate Format: {'Sexagesimal (RA/Dec only)' if use_sexagesimal else 'Decimal'}\n")
                                f.write(f"# Generated: {datetime.now(utc).strftime('%Y-%m-%d %H:%M:%S')} UTC\n\n")
                           
                            df.to_csv(filename, index=False, mode='a')
               
                messagebox.showinfo("Success", "Look angles generated successfully")
           
            ttk.Button(dialog, text="Generate", command=process_look_angles).pack(pady=5)
           
        except Exception as e:
            messagebox.showerror("Error", str(e))
       
    def load_tle(self):
        try:
            filename = filedialog.askopenfilename(
                filetypes=[("TLE files", "*.txt"), ("All files", "*.*")])
            if not filename:
                return
           
            # Load satellites
            satellites = self.predictor.load_tle_file(filename)
           
            # Calculate visibility for all satellites
            self.all_candidates = self.predictor.find_best_satellites(
                satellites=satellites,
                current_time=self.current_time,
                duration_hours=48
            )
           
            # Update display
            self.update_table()
           
            messagebox.showinfo("Success",
                              f"Loaded {len(self.all_candidates)} satellites")
           
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def save_to_csv(self):
        try:
            # Auto-generate filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            default_filename = f"visibility_{timestamp}.csv"
           
            filename = filedialog.asksaveasfilename(
                initialfile=default_filename,
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            if not filename:
                return
           
            # Get all items from treeview
            data = []
            for item in self.tree.get_children():
                values = self.tree.item(item)['values']
               
                # Get speed unit from current selection
                speed_unit = self.speed_unit_var.get()
                speed_column_name = "Angular_Speed_arcmin_per_sec" if speed_unit == "arcmin/s" else "Angular_Speed_deg_per_hour"
               
                data.append({
                    'Satellite': values[0],
                    'NORAD_ID': values[1],
                    'Semi_major_axis_km': values[2],
                    'Total_number_of_passes': values[3],
                    'Duration_min': values[4],
                    'Max_Elevation_deg': values[5],
                    speed_column_name: values[6],
                    'FOV_Duration_min': values[7],
                    'Sunlit': values[8],
                    'Night_Passes': values[9]
                })
           
            pd.DataFrame(data).to_csv(filename, index=False)
            messagebox.showinfo("Success", "Data saved successfully")
           
        except Exception as e:
            messagebox.showerror("Error", str(e))
           
    def sort_treeview(self, col):
        """Sort treeview when column header is clicked."""
        l = [(self.tree.set(k, col), k) for k in self.tree.get_children('')]
        try:
            l.sort(key=lambda t: float(t[0].split()[0]))
        except ValueError:
            l.sort()
           
        for index, (val, k) in enumerate(l):
            self.tree.move(k, '', index)
           
        # Switch the heading so that it will sort in the opposite direction next time
        self.tree.heading(col,
            command=lambda: self.sort_treeview_reverse(col))
           
    def sort_treeview_reverse(self, col):
        """Sort treeview in reverse order when column header is clicked again."""
        l = [(self.tree.set(k, col), k) for k in self.tree.get_children('')]
        try:
            l.sort(key=lambda t: float(t[0].split()[0]), reverse=True)
        except ValueError:
            l.sort(reverse=True)
           
        for index, (val, k) in enumerate(l):
            self.tree.move(k, '', index)
           
        self.tree.heading(col,
            command=lambda: self.sort_treeview(col))

    def apply_filters(self):
        """Apply current filters and update display."""
        self.update_table()

def main():
    root = tk.Tk()
    root.title("Satellite Visibility Predictor - ARIES")
   
    # Set smaller window size
    root.minsize(1200, 600)  # Reduced from 800 to 600
   
    # Configure style
    style = ttk.Style()
    style.theme_use('clam')
   
    app = SatelliteVisibilityGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
