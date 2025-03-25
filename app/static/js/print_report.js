/**
 * Print Report JavaScript
 * Handles the printing of medical reports directly in the browser
 */

function printMedicalReport(scanId) {
    // Open the print version of the report in a new window
    const printWindow = window.open(`/scans/${scanId}/print_report`, '_blank');
    
    // Wait for the window to load and then print
    printWindow.addEventListener('load', function() {
        // Wait a bit to ensure all resources are loaded
        setTimeout(() => {
            printWindow.print();
            // Some browsers may close the window after printing
            // Others may keep it open, which is useful for user to save as PDF if needed
        }, 500);
    });
}

// Add event listener to print button if it exists on page load
document.addEventListener('DOMContentLoaded', function() {
    const printButton = document.getElementById('print-report-btn');
    if (printButton) {
        printButton.addEventListener('click', function(e) {
            e.preventDefault();
            const scanId = this.getAttribute('data-scan-id');
            printMedicalReport(scanId);
        });
    }
}); 