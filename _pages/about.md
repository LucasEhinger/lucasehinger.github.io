---
permalink: /
title: "About me"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

I am a 4th year PhD student at MIT, studying experimental nuclear physics. When not fighting with computing clusters or performing percussive maintenance, I'm usually in the [mountains](/outdoors) or on the [water](/outdoors).  

I am an _\<insert positive adjective(s) of your choice\>_ individual; I give weekly presentations to 30-person collaborations as well as 1-1 and small group technical meetings. I'll take potentially experiment-ending complications over monotony any day of the week.

## [Current Research](/research/LAD)

<div style="display: flex; gap: 20px;">
<div style="flex: 1;">

I'm currently working under Prof. Or Hen, studying nuclear structure and the EMC Effect. This work required a year of data taking at Thomas Jefferson National Accelerator Facility, where (as the only graduate student on the experiment), I performed a wide variety of tasks ranging from detector and readout electronics installation and testing; data acquisition setup (triggers, FADC's, TDC's, etc.); raw signal processing; software development (C++); and higher level physics analysis. Having successfully collected all data in July 2025, I'm back at MIT currently working with a team of post-docs, research scientists, and professors to calibrate, analyze, and publish our data by 2027.

</div>
<img id="rotating-image" src="/images/research/LAD/LAD_1.jpeg" alt="Research" style="width: 300px; height: 300px; object-fit: cover;">
</div>

<script>
const images = ['/images/research/LAD/LAD_1.jpeg', '/images/research/LAD/LAD_2.jpeg', '/images/research/LAD/LAD_4.jpeg', '/images/research/LAD/LAD_5.jpeg', '/images/research/LAD/LAD_6.jpeg'];
let currentIndex = 0;
setInterval(() => {
  currentIndex = (currentIndex + 1) % images.length;
  document.getElementById('rotating-image').src = images[currentIndex];
}, 5000); // Changed to 5 seconds
</script>