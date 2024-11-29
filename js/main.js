const firebaseConfig = {
	apiKey: "AIzaSyBl1afesRYWwwfjI5hIFdjQM0o1GHIOspo",
	authDomain: "senior2-fa5a1.firebaseapp.com",
	projectId: "senior2-fa5a1",
	storageBucket: "senior2-fa5a1.appspot.com",
	messagingSenderId: "234868067519",
	appId: "1:234868067519:web:5ce36a09f758d42b7e0a0a",
};

const firebaseApp = firebase.initializeApp(firebaseConfig);

const auth = firebaseApp.auth();
const db = firebaseApp.firestore();

function signupUser(email, password, fname, lname) {
	clearResults();
	auth.createUserWithEmailAndPassword(email, password)
		.then(() => createUserProfile(fname, lname))
		.catch((err) => {
			if (err.message.includes("email-already-in-use")) {
				setErrorMessage("Email Already exists");
			} else if (err.message.includes("auth/weak-password")) {
				setErrorMessage("Password should be 6 characters at least");
			}
		});
}

function logout() {
	auth.signOut();
}

function truncate(str, n) {
	return str.length > n ? str.substr(0, n - 1) + "..." : str;
}

function createUserProfile(fname, lname) {
	db.collection("users")
		.doc(auth.currentUser.uid)
		.set({
			firstName: fname,
			lastName: lname,
			email: auth.currentUser.email,
		})
		.then(() => {
			window.location.href = "/";
		});
}

function login(email, password) {
	clearResults();
	return auth
		.signInWithEmailAndPassword(email, password)
		.then(() => {
			window.location.href = "/";
		})
		.catch((e) => {
			if (e.message.includes("auth/invalid-credential")) {
				throw new Error("Invalid login data");
			}
		});
}

function resetPassword(email) {
	clearResults();
	auth.sendPasswordResetEmail(email)
		.then(() => {
			setSuccessMessage("Password reset email sent");
		})
		.catch((e) => {
			setErrorMessage(e.message);
		});
}

function closeDetails() {
	document.getElementById("history-details").style.display = "none";
}

async function showDetails(id) {
	let data = await db.collection("history").doc(id).get();
	data = {
		id: data.id,
		...data.data(),
	};

	// Map Firestore fields to desired keys
	const orderedData = {
		jobTitle: data.job_title, // Adjust casing to match Firestore
		companyName: data.company_name, // Adjust casing
		location: data.location,
		education: data.education,
		jobDescription: data.job_description, // Adjust casing
		requirements: data.requirements,
		function: data.job_function, // Adjust casing to match Firestore
		benefit: data.benefits, // Adjust casing to match Firestore
		other: data.other_info, // Adjust casing to match Firestore
		url: data.url,
	};

	let container = document.getElementById("history-details-content");
	container.innerHTML = Object.entries(orderedData)
		.map(([key, value]) => {
			return `
            <div class="details-card">
                <span class="details-text">${key
					.replace(/([A-Z])/g, " $1")
					.replace(/^./, (str) => str.toUpperCase())}:</span>
                <span class="details-value ${!value ? "missing" : ""}">${value || "N/A"}</span>
            </div>
        `;
		})
		.join("");

	document.getElementById("history-details").style.display = "flex";
}
async function getHistory() {
	// date, url, result, image?
	const data = await db.collection("history").where("uid", "==", auth.currentUser.uid).get();
	const docs = data.docs.map((d) => ({ id: d.id, ...d.data() }));
	// sort based on date
	docs.sort((a, b) => b.date.toDate() - a.date.toDate());

	const container = document.getElementById("history-tbody");
	container.innerHTML = "";
	if (docs.length == 0) {
		document.getElementById("no-history").style.display = "inline-block";
		return;
	}
	const rows = docs.map(
		(d) => `
			<tr>
				<td class='truncate-td'>${d.date.toDate().toLocaleString()}</td>
				<td class='truncate-td'>
					<a target="_blank" href="${d.url}">${truncate(d.url, 15)}</a>
				</td>
				<td class='truncate-td'>
					${d.url_result}
				</td>
				<td class='truncate-td'>
					<a onclick="showDetails('${d.id}')" href="#">show deatils</a>
				</td>
				<td class='truncate-td'>
					${d.type == "job" ? d.job_result : "-"}
				</td>
			</tr>
		`
	);

	container.innerHTML = rows.join("");
}

function getProfileData() {
	db.collection("users")
		.doc(auth.currentUser.uid)
		.get()
		.then((d) => {
			const data = d.data();
			document.getElementById("FN").value = data.firstName;
			document.getElementById("LN").value = data.lastName;
			document.getElementById("Email").value = data.email;
		});
}

function updateProfile(firstName, lastName) {
	db.collection("users")
		.doc(auth.currentUser.uid)
		.update({
			firstName,
			lastName,
		})
		.then(() => {
			setSuccessMessage("Profile updated");
			getProfileData();
		});
}

function setSuccessMessage(message) {
	document.getElementById("result-success").innerHTML = message;
}

function setErrorMessage(message) {
	document.getElementById("result-success").innerHTML = message;
}

function clearResults() {
	setSuccessMessage("");
	setErrorMessage("");
}

function updateHtmlElementsBasedOnAuthState() {
	if (auth.currentUser) {
		try {
			let lastUrlResultStr = localStorage.getItem("lastUrlResult");
			if (lastUrlResultStr) {
				let lastUrlResult = JSON.parse(lastUrlResultStr);
				$("#url").val(lastUrlResult.url);
				$("#url-result").val(lastUrlResult.url_result);
			}
			$("#uid").val(auth.currentUser.uid);
			$(".only-visitor").remove();
		} catch (e) {}
	} else {
		$(".only-user").remove();
	}
}

auth.onAuthStateChanged((user) => {
	updateHtmlElementsBasedOnAuthState();
});

async function saveUrlHistory(url, result, uid) {
	let historyItem = { url, url_result: result, date: new Date(), type: "url", uid };
	localStorage.setItem("lastUrlResult", JSON.stringify(historyItem));
	if (!result.toLowerCase().includes("safe")) {
		await pushHistoryItem(historyItem);
	}
}

async function saveJobHistory(historyItem) {
	historyItem = {
		...historyItem,
		date: new Date(),
		type: "job",
	};
	await pushHistoryItem(historyItem);
}

async function pushHistoryItem(historyItem) {
	if (!historyItem.uid || historyItem.uid.trim() === "") {
		return;
	}
	return db.collection("history").add(historyItem);
}
