document.addEventListener("DOMContentLoaded", () => {
    const bed_id = "{{ bed_id }}"; // Pastikan variabel ini diterima dari backend
    fetchVitalData(bed_id);
});
