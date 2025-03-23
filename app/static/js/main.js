// Update WebSocket connection
const wsUrl = `ws://${window.location.hostname}:8080/display`;
const socket = new WebSocket(wsUrl);

socket.onopen = () => {
    console.log('Connected to WebSocket server');
};

socket.onmessage = (event) => {
    const data = JSON.parse(event.data);
    updateCharts(data.data);
    updateActivityStatus(data.activity, data.confidence);
};

socket.onerror = (error) => {
    console.error('WebSocket error:', error);
};

socket.onclose = () => {
    console.log('Disconnected from WebSocket server');
};

let accChart, gyroChart;
const dataPoints = 50;
const initialData = Array(dataPoints).fill({ ax: 0, ay: 0, az: 0, gx: 0, gy: 0, gz: 0 });

// Chart configuration
const chartConfig = {
    type: 'line',
    options: {
        responsive: true,
        animation: false,
        scales: {
            y: {
                min: -20,
                max: 20
            }
        }
    }
};

// Initialize charts when page loads
document.addEventListener('DOMContentLoaded', () => {
    initializeCharts();
});

function initializeCharts() {
    // Initialize Accelerometer Chart
    const accCtx = document.getElementById('accChart').getContext('2d');
    accChart = new Chart(accCtx, {
        ...chartConfig,
        data: createChartData('Acc', initialData)
    });

    // Initialize Gyroscope Chart
    const gyroCtx = document.getElementById('gyroChart').getContext('2d');
    gyroChart = new Chart(gyroCtx, {
        ...chartConfig,
        data: createChartData('Gyro', initialData)
    });
}

function createChartData(type, data) {
    const labels = Array(dataPoints).fill('');
    return {
        labels,
        datasets: ['X', 'Y', 'Z'].map((axis, i) => ({
            label: `${type}${axis}`,
            data: data.map(p => p[`${type.toLowerCase()}${axis.toLowerCase()}`]),
            borderColor: getColor(type, i),
            tension: 0.2,
            pointRadius: 0
        }))
    };
}

function getColor(type, index) {
    const colors = {
        Acc: ['#FF6B6B', '#4ECDC4', '#FFE66D'],
        Gyro: ['#6BFFB8', '#FF9F6B', '#9B6BFF']
    };
    return colors[type][index];
}

function updateCharts(data) {
    accChart.data.datasets.forEach((dataset, i) => {
        dataset.data.shift();
        dataset.data.push(data[`a${['x', 'y', 'z'][i]}`]);
    });
    accChart.update('none');

    gyroChart.data.datasets.forEach((dataset, i) => {
        dataset.data.shift();
        dataset.data.push(data[`g${['x', 'y', 'z'][i]}`]);
    });
    gyroChart.update('none');
}

function updateActivityStatus(activity, confidence) {
    document.getElementById('activity').textContent = activity?.toUpperCase() || 'UNKNOWN';
    document.getElementById('confidence').textContent = 
        `Confidence: ${((confidence || 0) * 100).toFixed(1)}%`;
}

function updateSensorGrid(data) {
    const sensorGrid = document.getElementById('sensorGrid');
    sensorGrid.innerHTML = '';

    const sensors = [
        ['AccX', 'ax'], ['AccY', 'ay'], ['AccZ', 'az'],
        ['GyroX', 'gx'], ['GyroY', 'gy'], ['GyroZ', 'gz']
    ];

    sensors.forEach(([label, key]) => {
        const div = document.createElement('div');
        div.className = 'sensor-value';
        div.innerHTML = `
            <span class="sensor-label">${label}:</span>
            <span class="sensor-number">${data[key].toFixed(2)}</span>
        `;
        sensorGrid.appendChild(div);
    });
}