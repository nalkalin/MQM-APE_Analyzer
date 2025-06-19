import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional
import os
from collections import Counter, defaultdict
import warnings

warnings.filterwarnings('ignore')

# Configure font support for better visualization
plt.rcParams['font.serif'] = ['DejaVu Serif', 'Times New Roman', 'serif']
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.unicode_minus'] = False

# Set style for better visualization
plt.style.use('seaborn-v0_8-whitegrid')

# Customized color palettes with distinct inner/outer schemes
SEVERITY_COLORS = {
    'Critical': '#D32F2F',  # red
    'Major': '#F57C00',  # orange
    'Minor': '#1976D2'  # blue
}

# Customized color for inner ring - pink/cyan/magenta tones
CATEGORY_COLORS = {
    'accuracy': '#E91E63',  # Pink
    'fluency': '#00BCD4',  # Cyan
    'style': '#8E24AA',  # Magenta
    'terminology': '#FFC107',  # Yellow
    'other': '#FF5722'  # Deep orange
}


class MQMAPEAnalyzer:
    def __init__(self, json_file_path: str, segments_per_task: int = 4):
        """
        Initialize MQM-APE Analyzer with score normalization

        Args:
            json_file_path: Path to JSON file
            segments_per_task: Number of segments per task for normalization (default: 4,
                             can be modified based on specific translation task requirements)
        """
        self.json_file_path = json_file_path
        self.segments_per_task = segments_per_task
        self.data = self._load_data()
        self.severity_mapping = {
            'critical': 'Critical',
            'major': 'Major',
            'minor': 'Minor'
        }

    def _load_data(self) -> List[Dict[str, Any]]:
        """
        Safely load JSON data

        Returns:
            Parsed JSON data
        """
        if not os.path.exists(self.json_file_path):
            raise FileNotFoundError(f"File not found: {self.json_file_path}")

        try:
            with open(self.json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if not isinstance(data, list):
                    raise ValueError("JSON file format error: Expected root element to be a list")
                return data
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON parsing error: {e}")
        except Exception as e:
            raise Exception(f"Error occurred while reading file: {e}")

    def calculate_normalized_scores(self) -> Dict[str, Any]:
        """
        Calculate normalized MQM scores following the methodology:
        1. Calculate number of students based on total_segments / segments_per_task
        2. Calculate normalized penalty score for each student (task)
        3. Scale scores to 0-100 based on all students' performance

        Returns:
            Dictionary containing normalized scores and scaling information
        """
        total_segments = len(self.data)
        num_students = total_segments // self.segments_per_task

        if num_students == 0:
            # Handle case where we have fewer segments than segments_per_task
            num_students = 1
            effective_segments_per_task = total_segments
        else:
            effective_segments_per_task = self.segments_per_task

        # Calculate normalized penalty scores for each student (task)
        student_normalized_penalties = []

        for student_idx in range(num_students):
            start_idx = student_idx * effective_segments_per_task
            end_idx = min(start_idx + effective_segments_per_task, total_segments)

            # Extract MQM scores for this student's segments
            student_scores = []
            for idx in range(start_idx, end_idx):
                student_scores.append(self.data[idx].get('MQM_APE_score', 0))

            # Calculate normalized penalty for this student
            # MQM scores are negative penalties, convert to positive for calculation
            penalties = [-score for score in student_scores]
            normalized_penalty = sum(penalties) / len(penalties) if len(penalties) > 0 else 0
            student_normalized_penalties.append(normalized_penalty)

        # Calculate min and max across all students for scaling
        min_penalty = min(student_normalized_penalties) if student_normalized_penalties else 0
        max_penalty = max(student_normalized_penalties) if student_normalized_penalties else 0

        # Calculate scaled quality scores (0-100) for each student
        student_scaled_scores = []
        for penalty in student_normalized_penalties:
            if max_penalty != min_penalty:
                scaled_score = 100 * (max_penalty - penalty) / (max_penalty - min_penalty)
            else:
                scaled_score = 100  # Perfect score when no variation
            student_scaled_scores.append(scaled_score)

        # For the current data, use the first student's scores as representative
        current_normalized_penalty = student_normalized_penalties[0] if student_normalized_penalties else 0
        current_scaled_score = student_scaled_scores[0] if student_scaled_scores else 100

        return {
            'total_segments': total_segments,
            'num_students': num_students,
            'segments_per_task': effective_segments_per_task,
            'student_normalized_penalties': student_normalized_penalties,
            'student_scaled_scores': student_scaled_scores,
            'current_normalized_penalty': current_normalized_penalty,
            'current_scaled_score': current_scaled_score,
            'min_penalty': min_penalty,
            'max_penalty': max_penalty
        }

    def extract_errors(self) -> List[Dict[str, Any]]:
        """
        Extract all error information, including entries without errors

        Returns:
            List of dictionaries containing error information
        """
        errors_list = []

        for idx, item in enumerate(self.data):
            source_text = item.get('source_seg', '')
            target_text = item.get('target_seg', '')
            source_lang = item.get('source_lang', '')
            target_lang = item.get('target_lang', '')
            overall_score = item.get('MQM_APE_score', 0)

            error_dict = item.get('error_dict', {})
            has_errors = False

            # Process different error severities
            for severity in ['critical', 'major', 'minor']:
                errors = error_dict.get(severity, [])

                for error in errors:
                    has_errors = True
                    error_info = {
                        'Item Number': idx + 1,
                        'Source Language': source_lang,
                        'Target Language': target_lang,
                        'Source Text': source_text,
                        'Translation Text': target_text,
                        'Original Score': overall_score,
                        'Error Severity': self.severity_mapping.get(severity, severity),
                        'Error Type': error.get('category', 'Unknown'),
                        'Error Span': error.get('span', ''),
                        'Post-Edit Text': error.get('post_edit', '')
                    }
                    errors_list.append(error_info)

            # If no errors, add a record
            if not has_errors:
                error_info = {
                    'Item Number': idx + 1,
                    'Source Language': source_lang,
                    'Target Language': target_lang,
                    'Source Text': source_text,
                    'Translation Text': target_text,
                    'Original Score': overall_score,
                    'Error Severity': 'No Error',
                    'Error Type': 'No Error',
                    'Error Span': '',
                    'Post-Edit Text': ''
                }
                errors_list.append(error_info)

        return errors_list

    def prepare_nested_pie_data(self):
        """
        Prepare data for nested pie chart with severity (outer) and categories (inner)
        """
        try:
            errors = self.extract_errors()
            df = pd.DataFrame(errors)

            if df.empty or df[df['Error Severity'] != 'No Error'].empty:
                return None, None, None, None, None, None

            error_df = df[df['Error Severity'] != 'No Error']

            # Group by severity and category
            severity_category_counts = defaultdict(lambda: defaultdict(int))
            severity_totals = defaultdict(int)

            for _, row in error_df.iterrows():
                severity = row['Error Severity']
                error_type = row['Error Type']

                # Extract main category from error type (e.g., "accuracy/mistranslation" -> "accuracy")
                main_category = error_type.split('/')[0] if '/' in error_type else 'other'

                severity_category_counts[severity][main_category] += 1
                severity_totals[severity] += 1

            # Prepare data for outer ring (severity)
            severity_order = ['Critical', 'Major', 'Minor']
            outer_sizes = []
            outer_labels = []
            outer_colors = []

            # Prepare data for inner ring (categories)
            inner_sizes = []
            inner_labels = []
            inner_colors = []

            for severity in severity_order:
                if severity in severity_totals:
                    outer_sizes.append(severity_totals[severity])
                    outer_labels.append(severity)
                    outer_colors.append(SEVERITY_COLORS[severity])

                    # Add categories for this severity
                    for category, count in severity_category_counts[severity].items():
                        inner_sizes.append(count)
                        inner_labels.append(category)
                        # Use category color with some variation based on severity
                        if category in CATEGORY_COLORS:
                            inner_colors.append(CATEGORY_COLORS[category])
                        else:
                            inner_colors.append(CATEGORY_COLORS['other'])

            return outer_sizes, outer_labels, outer_colors, inner_sizes, inner_labels, inner_colors

        except Exception as e:
            print(f"Error in prepare_nested_pie_data: {e}")
            return None, None, None, None, None, None

    def create_visualizations(self, save_path: str = "mqm_charts.png"):
        """
        Create visualization charts

        Args:
            save_path: Path to save charts
        """
        try:
            errors = self.extract_errors()
            df = pd.DataFrame(errors)
            score_info = self.calculate_normalized_scores()

            # Create figure with better layout - 2x2 grid
            fig, axes = plt.subplots(2, 2, figsize=(18, 14))
            fig.suptitle('MQM-APE Translation Quality Analysis',
                         fontsize=18, fontweight='bold', y=0.98)

            # 1. Nested Pie Chart - Error Distribution (top-left)
            pie_data = self.prepare_nested_pie_data()
            if pie_data is not None and len(pie_data) == 6:
                outer_sizes, outer_labels, outer_colors, inner_sizes, inner_labels, inner_colors = pie_data
            else:
                outer_sizes = outer_labels = outer_colors = inner_sizes = inner_labels = inner_colors = None

            if outer_sizes is not None:
                # Create nested pie chart with external labels for outer ring
                size = 0.3

                # Outer ring (severity) with external labels
                wedges1, texts1 = axes[0, 0].pie(
                    outer_sizes, colors=outer_colors,
                    radius=1, startangle=90,
                    wedgeprops=dict(width=size, edgecolor='white', linewidth=2),
                    labels=[f'{label}\n{count} ({count / sum(outer_sizes) * 100:.1f}%)'
                            for label, count in zip(outer_labels, outer_sizes)],
                    labeldistance=1.23
                )

                # Inner ring (categories) - no labels on the pie
                wedges2, _ = axes[0, 0].pie(
                    inner_sizes, colors=inner_colors,
                    radius=1 - size, startangle=90,
                    wedgeprops=dict(width=size, edgecolor='white', linewidth=1)
                )

                # Style the outer ring labels
                for text in texts1:
                    text.set_fontsize(10)
                    text.set_fontweight('bold')
                    text.set_ha('center')

                # Create legend only for inner ring (categories)
                category_legend_elements = []
                category_counts = {}
                for i, (label, color) in enumerate(zip(inner_labels, inner_colors)):
                    count = inner_sizes[i]
                    if label not in category_counts:
                        category_counts[label] = 0
                    category_counts[label] += count

                for label, total_count in category_counts.items():
                    color = CATEGORY_COLORS.get(label, CATEGORY_COLORS['other'])
                    percentage = (total_count / sum(inner_sizes)) * 100
                    category_legend_elements.append(
                        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=color,
                                   markersize=10, label=f'{label}: {total_count} ({percentage:.1f}%)')
                    )

                axes[0, 0].legend(handles=category_legend_elements, loc='center left',
                                  bbox_to_anchor=(1.1, 0.5), fontsize=10, title='Categories')

                axes[0, 0].set_title('Error Distribution by Severity & Category\n(Outer: Severity, Inner: Category)',
                                     fontsize=14, fontweight='bold', pad=20)
            else:
                axes[0, 0].text(0.5, 0.5, 'No Errors Found', ha='center', va='center',
                                fontsize=16, fontweight='bold', transform=axes[0, 0].transAxes)
                axes[0, 0].set_title('Error Distribution', fontsize=14, fontweight='bold', pad=20)

            # 2. Error Type Distribution (top-right)
            if not df.empty and not df[df['Error Severity'] != 'No Error'].empty:
                error_df = df[df['Error Severity'] != 'No Error']
                category_counts = error_df['Error Type'].value_counts().head(10)

                # colors for error types
                type_colors = ['#D32F2F', '#1976D2', '#388E3C', '#F57C00', '#7B1FA2',
                               '#00796B', '#C2185B', '#5D4037', '#455A64', '#E64A19']

                bars = axes[0, 1].barh(
                    range(len(category_counts)),
                    category_counts.values,
                    color=type_colors[:len(category_counts)],
                    alpha=0.8,
                    edgecolor='white',
                    linewidth=1
                )
                axes[0, 1].set_yticks(range(len(category_counts)))
                axes[0, 1].set_yticklabels(category_counts.index, fontsize=10)
                axes[0, 1].set_title('Error Type Distribution (Top 10)', fontsize=14, fontweight='bold', pad=20)
                axes[0, 1].set_xlabel('Count', fontsize=12)
                axes[0, 1].grid(axis='x', alpha=0.3, linestyle='--')

                # Add value labels on bars
                for i, bar in enumerate(bars):
                    width = bar.get_width()
                    axes[0, 1].text(width + max(category_counts.values) * 0.02,
                                    bar.get_y() + bar.get_height() / 2,
                                    f'{int(width)}', ha='left', va='center',
                                    fontweight='bold', fontsize=9)
            else:
                axes[0, 1].text(0.5, 0.5, 'No Errors Found', ha='center', va='center',
                                fontsize=14, fontweight='bold', transform=axes[0, 1].transAxes)
                axes[0, 1].set_title('Error Type Distribution', fontsize=14, fontweight='bold', pad=20)

            # 3. Score Distribution (bottom-left)
            scores = [item.get('MQM_APE_score', 0) for item in self.data]
            n, bins, patches = axes[1, 0].hist(
                scores, bins=max(8, min(15, len(set(scores)))), alpha=0.8,
                color='#1976D2', edgecolor='#0D47A1', linewidth=2
            )
            axes[1, 0].set_title('Original MQM Score Distribution', fontsize=14, fontweight='bold', pad=20)
            axes[1, 0].set_xlabel('MQM Score', fontsize=12)
            axes[1, 0].set_ylabel('Frequency', fontsize=12)

            # Add mean line
            mean_score = np.mean(scores)
            axes[1, 0].axvline(mean_score, color='#D32F2F', linestyle='--', linewidth=3,
                               label=f'Mean: {mean_score:.2f}', alpha=0.9)
            axes[1, 0].legend(fontsize=11, frameon=True, fancybox=True, shadow=True)
            axes[1, 0].grid(alpha=0.3, linestyle='--')

            # Apply gradient color to histogram bars
            for i, patch in enumerate(patches):
                patch.set_facecolor(plt.cm.Blues(0.4 + 0.6 * i / len(patches)))

            # 4. Quality Score Box Plot (bottom-right)
            all_scaled_scores = score_info['student_scaled_scores']

            if len(all_scaled_scores) > 1:
                box_parts = axes[1, 1].boxplot(
                    all_scaled_scores,
                    patch_artist=True,
                    boxprops=dict(facecolor='#2E86AB', alpha=0.7, linewidth=1.5),
                    medianprops=dict(color='#1F4E79', linewidth=2.5),
                    whiskerprops=dict(color='#1F4E79', linewidth=1.5),
                    capprops=dict(color='#1F4E79', linewidth=1.5),
                    flierprops=dict(marker='o', markerfacecolor='#A23B72', markersize=8, alpha=0.8,
                                    markeredgecolor='#1F4E79', markeredgewidth=1)
                )

                # Add individual points overlay
                y_positions = np.random.normal(1, 0.04, len(all_scaled_scores))
                axes[1, 1].scatter(y_positions, all_scaled_scores,
                                   color='#34495E', alpha=0.6, s=50, zorder=3, edgecolors='#1F4E79', linewidth=1)
            else:
                # If only one student, show as a single point
                axes[1, 1].scatter([1], all_scaled_scores, color='#2E86AB', s=200,
                                   alpha=0.8, edgecolors='#1F4E79', linewidth=2, zorder=3)

            axes[1, 1].set_title('Quality Score Distribution\n(0-100 Scale)', fontsize=14, fontweight='bold', pad=20)
            axes[1, 1].set_ylabel('Scaled Quality Score', fontsize=12)
            axes[1, 1].set_xticklabels(['Translation Quality'])
            axes[1, 1].grid(alpha=0.3, linestyle='--')
            axes[1, 1].set_ylim(0, 105)

            # Add score annotation
            annotation_text = (f'Students: {score_info["num_students"]}\n'
                               f'Current Score: {score_info["current_scaled_score"]:.1f}/100\n'
                               f'Penalty per Segment: {score_info["current_normalized_penalty"]:.2f}')

            axes[1, 1].text(0.02, 0.98, annotation_text,
                            transform=axes[1, 1].transAxes, fontsize=11,
                            verticalalignment='top',
                            bbox=dict(boxstyle='round,pad=0.6', facecolor='#ECEFF1',
                                      alpha=0.95, edgecolor='#90A4AE', linewidth=1))

            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.show()
            print(f"Visualization charts saved to: {save_path}")

        except Exception as e:
            print(f"Error occurred while creating visualizations: {e}")
            import traceback
            traceback.print_exc()

    def create_summary_table(self) -> pd.DataFrame:
        """
        Create error statistics summary table

        Returns:
            DataFrame containing error statistics
        """
        errors = self.extract_errors()
        df = pd.DataFrame(errors)

        if df.empty:
            return pd.DataFrame()

        # Exclude no-error entries
        error_df = df[df['Error Severity'] != 'No Error']

        if error_df.empty:
            return pd.DataFrame(
                [{'Error Severity': 'None', 'Error Type': 'All translations are perfect', 'Count': 0}])

        # Group by error severity and type
        summary = error_df.groupby(['Error Severity', 'Error Type']).agg({
            'Item Number': 'count'
        }).reset_index()

        summary.columns = ['Error Severity', 'Error Type', 'Count']

        # Sort by error severity (Critical -> Major -> Minor)
        severity_order = ['Critical', 'Major', 'Minor']
        summary['Error Severity'] = pd.Categorical(summary['Error Severity'], categories=severity_order, ordered=True)
        summary = summary.sort_values(['Error Severity', 'Count'], ascending=[True, False])

        return summary

    def create_detailed_table(self) -> pd.DataFrame:
        """
        Create detailed error table with normalized scores grouped by segments

        Returns:
            DataFrame containing all detailed error information
        """
        errors = self.extract_errors()
        df = pd.DataFrame(errors)

        # Add normalized score information
        score_info = self.calculate_normalized_scores()

        if not df.empty:
            # Calculate error density (number of errors per translation)
            error_counts = df[df['Error Severity'] != 'No Error'].groupby('Item Number').size()
            df['Error Count for This Item'] = df['Item Number'].map(error_counts).fillna(0).astype(int)

            # Add normalized score information based on student grouping
            df['Student ID'] = ((df['Item Number'] - 1) // self.segments_per_task) + 1

            # Map normalized scores to appropriate students
            for idx, row in df.iterrows():
                student_id = row['Student ID']
                if student_id <= len(score_info['student_normalized_penalties']):
                    df.at[idx, 'Normalized Penalty Score'] = score_info['student_normalized_penalties'][student_id - 1]
                    df.at[idx, 'Scaled Quality Score (0-100)'] = score_info['student_scaled_scores'][student_id - 1]
                else:
                    df.at[idx, 'Normalized Penalty Score'] = 0
                    df.at[idx, 'Scaled Quality Score (0-100)'] = 100

        return df

    def export_to_excel(self, output_file: str = 'mqm_analysis.xlsx'):
        """
        Export analysis results to Excel file with score normalization

        Args:
            output_file: Output Excel filename
        """
        try:
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                # 1. Detailed error table with normalized scores grouped properly
                detailed_df = self.create_detailed_table()
                # Remove unnecessary columns and reorder
                columns_to_remove = ['Error Count for This Item']
                detailed_df_clean = detailed_df.drop(
                    columns=[col for col in columns_to_remove if col in detailed_df.columns])

                # Reorder columns for better readability
                if not detailed_df_clean.empty:
                    column_order = ['Student ID', 'Item Number', 'Source Language', 'Target Language',
                                    'Source Text', 'Translation Text', 'Original Score',
                                    'Normalized Penalty Score', 'Scaled Quality Score (0-100)',
                                    'Error Severity', 'Error Type', 'Error Span', 'Post-Edit Text']
                    # Only include columns that exist
                    existing_columns = [col for col in column_order if col in detailed_df_clean.columns]
                    detailed_df_clean = detailed_df_clean[existing_columns]

                detailed_df_clean.to_excel(writer, sheet_name='Detailed Analysis', index=False)

                # 2. Error statistics summary
                summary_df = self.create_summary_table()
                summary_df.to_excel(writer, sheet_name='Error Summary', index=False)

                # 3. Overall statistics with normalized scores
                overall_stats = self.get_statistics()
                stats_df = pd.DataFrame(list(overall_stats.items()), columns=['Metric', 'Value'])
                stats_df.to_excel(writer, sheet_name='Quality Metrics', index=False)

                # 4. Score normalization details
                score_info = self.calculate_normalized_scores()

                # Create a summary table for all students
                students_data = []
                for i, (penalty, scaled) in enumerate(zip(score_info['student_normalized_penalties'],
                                                          score_info['student_scaled_scores'])):
                    students_data.append({
                        'Student_ID': i + 1,
                        'Normalized_Penalty': penalty,
                        'Scaled_Quality_Score': scaled
                    })

                students_df = pd.DataFrame(students_data)
                students_df.to_excel(writer, sheet_name='Students Scores', index=False)

                # Normalization metadata
                norm_metadata = {
                    'Total_Segments': score_info['total_segments'],
                    'Number_of_Students': score_info['num_students'],
                    'Segments_per_Task': score_info['segments_per_task'],
                    'Min_Penalty': score_info['min_penalty'],
                    'Max_Penalty': score_info['max_penalty']
                }
                meta_df = pd.DataFrame([norm_metadata])
                meta_df.to_excel(writer, sheet_name='Normalization Info', index=False)

            print(f"Analysis results successfully saved to: {output_file}")
            print(f"Contains {len(pd.ExcelFile(output_file).sheet_names)} worksheets")

        except Exception as e:
            print(f"Error occurred while exporting to Excel: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics including normalized scores

        Returns:
            Dictionary containing comprehensive statistics
        """
        errors = self.extract_errors()
        df = pd.DataFrame(errors)
        score_info = self.calculate_normalized_scores()

        if df.empty:
            return {"Error": "No data"}

        error_df = df[df['Error Severity'] != 'No Error']
        scores = [item.get('MQM_APE_score', 0) for item in self.data]

        stats = {
            'Total Translation Segments': len(self.data),
            'Number of Students': score_info['num_students'],
            'Segments per Task': score_info['segments_per_task'],
            'Segments with Errors': len(error_df['Item Number'].unique()) if not error_df.empty else 0,
            'Segments without Errors': len(self.data) - (
                len(error_df['Item Number'].unique()) if not error_df.empty else 0),
            'Total Error Count': len(error_df),
            'Average Errors per Segment': len(error_df) / len(self.data) if len(self.data) > 0 else 0,
            'Original Average MQM Score': np.mean(scores),
            'Current Normalized Penalty Score': score_info['current_normalized_penalty'],
            'Current Scaled Quality Score (0-100)': score_info['current_scaled_score'],
            'Min Penalty Across Students': score_info['min_penalty'],
            'Max Penalty Across Students': score_info['max_penalty'],
            'Highest Original Score': max(scores),
            'Lowest Original Score': min(scores),
            'Score Standard Deviation': np.std(scores),
            'Critical Errors': len(error_df[error_df['Error Severity'] == 'Critical']) if not error_df.empty else 0,
            'Major Errors': len(error_df[error_df['Error Severity'] == 'Major']) if not error_df.empty else 0,
            'Minor Errors': len(error_df[error_df['Error Severity'] == 'Minor']) if not error_df.empty else 0
        }

        # Format numerical values
        for key, value in stats.items():
            if isinstance(value, float):
                stats[key] = round(value, 2)

        return stats

    def print_summary(self):
        """
        Print analysis summary with normalized scores
        """
        try:
            errors = self.extract_errors()
            if not errors:
                print("No data found")
                return

            df = pd.DataFrame(errors)
            error_df = df[df['Error Severity'] != 'No Error']
            score_info = self.calculate_normalized_scores()

            print("\n" + "=" * 60)
            print("MQM-APE Translation Quality Analysis")
            print("=" * 60)

            # Basic statistics
            stats = self.get_statistics()
            print(f"\nBasic Statistics:")
            print(f"   Total Translation Segments: {stats['Total Translation Segments']}")
            print(f"   Number of Students: {stats['Number of Students']}")
            print(f"   Segments per Task: {stats['Segments per Task']}")
            print(f"   Segments with Errors: {stats['Segments with Errors']}")
            print(f"   Segments without Errors: {stats['Segments without Errors']}")
            print(f"   Total Error Count: {stats['Total Error Count']}")
            print(f"   Average Errors per Segment: {stats['Average Errors per Segment']:.2f}")

            # Score analysis with normalization
            print(f"\nScore Analysis (Normalized):")
            print(f"   Original Average MQM Score: {stats['Original Average MQM Score']:.2f}")
            print(f"   Current Normalized Penalty Score: {stats['Current Normalized Penalty Score']:.2f}")
            print(f"   Current Scaled Quality Score (0-100): {stats['Current Scaled Quality Score (0-100)']:.1f}")
            print(f"   Score Range: {stats['Lowest Original Score']:.1f} ~ {stats['Highest Original Score']:.1f}")
            print(f"   Min Penalty Across Students: {stats['Min Penalty Across Students']:.2f}")
            print(f"   Max Penalty Across Students: {stats['Max Penalty Across Students']:.2f}")

            # All students' scores
            print(f"\nAll Students' Scaled Quality Scores:")
            for i, score in enumerate(score_info['student_scaled_scores']):
                print(f"   Student {i + 1}: {score:.1f}/100")

            if not error_df.empty:
                # Error severity statistics
                print(f"\nError Severity Statistics:")
                severity_count = error_df['Error Severity'].value_counts()
                for severity, count in severity_count.items():
                    percentage = (count / len(error_df)) * 100
                    print(f"   {severity}: {count} errors ({percentage:.1f}%)")

                # Error type statistics (top 8)
                print(f"\nMost Common Error Types (Top 8):")
                category_count = error_df['Error Type'].value_counts().head(8)
                for i, (category, count) in enumerate(category_count.items(), 1):
                    percentage = (count / len(error_df)) * 100
                    print(f"   {i:2d}. {category}: {count} errors ({percentage:.1f}%)")
            else:
                print(f"\nCongratulations! No errors found in all translations!")

            print("=" * 60)

        except Exception as e:
            print(f"Error occurred while generating summary: {e}")


def main():
    """
    Main function demonstrating MQM-APE analysis with score normalization
    """
    json_file = "results.json"  # Replace with your JSON file path

    try:
        print("Starting MQM-APE Translation Quality Analyzer...")
        print("Note: Default segments per task = 4 (can be modified based on your translation task)")

        # Create analyzer
        # Modify segments_per_task parameter based on your specific translation task requirements
        analyzer = MQMAPEAnalyzer(json_file, segments_per_task=4)

        # Print summary with normalized scores
        analyzer.print_summary()

        # Create visualization charts
        print("\nGenerating visualization charts...")
        analyzer.create_visualizations()

        # Export to Excel with normalized score information
        print("\nExporting analysis results...")
        analyzer.export_to_excel()

        print("\nAnalysis Complete! Files generated:")
        print("   mqm_analysis.xlsx - Comprehensive analysis with normalized scores")
        print("   mqm_charts.png - Visualization")
        print("\nFor different translation tasks, modify 'segments_per_task' parameter in the analyzer initialization.")

    except FileNotFoundError:
        print(f"Error: File {json_file} not found")
        print("   Please ensure the JSON file path is correct")
    except ValueError as e:
        print(f"Data format error: {e}")
    except Exception as e:
        print(f"Error occurred during processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()