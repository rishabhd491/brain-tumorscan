/**
 * Brain Tumor MRI Classification App
 * Main JavaScript functionality
 */

document.addEventListener('DOMContentLoaded', function() {
    // Handle file uploads with preview
    setupFileUpload();
    
    // Setup DataTables if available
    setupDataTables();
    
    // Initialize popovers and tooltips
    initializeBootstrapComponents();
    
    // Toggle patient visibility on the patients page
    const toggleVisibilityButtons = document.querySelectorAll('.toggle-visibility');
    
    if (toggleVisibilityButtons.length > 0) {
        toggleVisibilityButtons.forEach(button => {
            button.addEventListener('click', function() {
                const patientId = this.getAttribute('data-patient-id');
                const icon = this.querySelector('i');
                const isVisible = icon.classList.contains('fa-eye');
                
                // Optimistic UI update
                if (isVisible) {
                    icon.classList.replace('fa-eye', 'fa-eye-slash');
                    this.setAttribute('title', 'Make visible');
                } else {
                    icon.classList.replace('fa-eye-slash', 'fa-eye');
                    this.setAttribute('title', 'Hide');
                }
                
                // Send request to server
                fetch(`/patients/${patientId}/toggle-visibility`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    }
                })
                .then(response => {
                    if (!response.ok) {
                        // Revert UI change on error
                        if (isVisible) {
                            icon.classList.replace('fa-eye-slash', 'fa-eye');
                            this.setAttribute('title', 'Hide');
                        } else {
                            icon.classList.replace('fa-eye', 'fa-eye-slash');
                            this.setAttribute('title', 'Make visible');
                        }
                        throw new Error('Network response was not ok');
                    }
                    return response.json();
                })
                .then(data => {
                    // Optional: show success message
                    console.log(data.message);
                })
                .catch(error => {
                    console.error('Error:', error);
                    // Optional: show error message to user
                });
            });
        });
    }
    
    // Patient search functionality 
    const searchInput = document.getElementById('patientSearch');
    if (searchInput) {
        searchInput.addEventListener('input', function() {
            const searchTerm = this.value.toLowerCase();
            const patientRows = document.querySelectorAll('tbody tr');
            
            patientRows.forEach(row => {
                const patientText = row.textContent.toLowerCase();
                if (patientText.includes(searchTerm)) {
                    row.style.display = '';
                } else {
                    row.style.display = 'none';
                }
            });
        });
    }
});

/**
 * Setup file upload functionality with preview
 */
function setupFileUpload() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const previewDiv = document.getElementById('preview');
    const analyzeBtn = document.getElementById('analyzeBtn');
    
    if (!uploadArea || !fileInput) return;
    
    // Click on upload area to trigger file input
    uploadArea.addEventListener('click', function() {
        fileInput.click();
    });
    
    // Handle file selection
    fileInput.addEventListener('change', function(e) {
        handleFileSelect(e.target.files);
    });
    
    // Drag and drop functionality
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
    });
    
    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, function() {
            uploadArea.classList.add('highlight');
        });
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, function() {
            uploadArea.classList.remove('highlight');
        });
    });
    
    uploadArea.addEventListener('drop', function(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFileSelect(files);
    });
    
    function handleFileSelect(files) {
        if (files.length > 0) {
            const file = files[0];
            
            // Validate file type
            if (!file.type.match('image/(jpeg|jpg|png)')) {
                showAlert('Please select a valid image file (JPG, JPEG, or PNG).', 'danger');
                return;
            }
            
            // Show preview
            const reader = new FileReader();
            reader.onload = function(e) {
                previewDiv.innerHTML = `
                    <div class="position-relative">
                        <img src="${e.target.result}" class="img-fluid rounded" alt="Image Preview" style="max-height: 200px;">
                        <button type="button" class="btn-close position-absolute top-0 end-0 bg-light rounded-circle m-1" id="removePreview"></button>
                    </div>
                `;
                
                // Enable the analyze button
                if (analyzeBtn) {
                    analyzeBtn.disabled = false;
                }
                
                // Add event listener to remove preview button
                document.getElementById('removePreview').addEventListener('click', function(e) {
                    e.stopPropagation();
                    previewDiv.innerHTML = '';
                    fileInput.value = '';
                    if (analyzeBtn) {
                        analyzeBtn.disabled = true;
                    }
                });
            };
            reader.readAsDataURL(file);
        }
    }
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
}

/**
 * Initialize DataTables for better table UI/UX if available
 */
function setupDataTables() {
    if (typeof $.fn.DataTable !== 'undefined') {
        $('.data-table').DataTable({
            responsive: true,
            language: {
                search: "_INPUT_",
                searchPlaceholder: "Search records..."
            },
            lengthMenu: [[10, 25, 50, -1], [10, 25, 50, "All"]],
            pageLength: 10
        });
    }
}

/**
 * Initialize Bootstrap components
 */
function initializeBootstrapComponents() {
    // Initialize popovers
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
    
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Initialize toasts
    const toastElList = [].slice.call(document.querySelectorAll('.toast'));
    toastElList.map(function (toastEl) {
        return new bootstrap.Toast(toastEl);
    });
}

/**
 * Show alert message
 * @param {string} message - The message to display
 * @param {string} type - Alert type (success, danger, warning, info)
 */
function showAlert(message, type = 'warning') {
    const alertContainer = document.getElementById('alertContainer');
    if (!alertContainer) return;
    
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.role = 'alert';
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    alertContainer.appendChild(alertDiv);
    
    // Auto dismiss after 5 seconds
    setTimeout(() => {
        alertDiv.classList.remove('show');
        setTimeout(() => {
            alertDiv.remove();
        }, 150);
    }, 5000);
} 