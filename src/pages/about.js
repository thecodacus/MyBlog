import React from "react"
import Layout from "../components/Layout"
import PageHeroName from "../components/PageHeroName"
import PageWrapper from "../components/PageWrapper"
import Seo from "../components/Seo"
export default function About() {
	return (
		<Layout>
			<Seo title="About" />
			<PageHeroName>About</PageHeroName>
			<PageWrapper>
				<h1>Intro</h1>
				<p>This is a blog about Machine learning, computer vision, web3 programming, and other things.</p>
				<p>I'm a software engineer based in India. I'm a fan of Robotics, Machine Learning, and Artificial Intelligence.</p>
				<p>
					In this blog, I will be posting about my journey in the field of Machine Learning and Artificial Intelligence. I will also be sharing my
					thoughts and experiences about the same.
				</p>
				<h1>Story</h1>
				<p>
					I started this blog in the year 2017. I was working is a software company. while it was a good learning experience for me, there were a lot
					of things that I wanted to do in various other topics, and I wanted to share my experiences with others. So this blog was born.
				</p>
				<h1>What about all "Decentralized Blog" and all that?</h1>

				<p>
					In the beginning, the blog was hosted on Digital Ocean powered by WordPress, while it was convenient and easy to set up and use. But it was
					not the best. As a hobby blogger I am a forgetful person and last month Dec 2020 I forgot to review my Digital Ocean account and my blog was
					deleted. I asked the support team if they have any backup as I was paying extra for backups. They said they don't have any backups. So I
					realized the lifespan of my blog is depended on the number of people who are controlling the server.
				</p>
				<p>
					I started to think about how to make my blog more secure and I started to look for a solution. at the same time, I was also learning about
					blockchain and web3. So I thought why not transform my blog into a decentralized blog. So I needed to make a blog that can run without a
					backend server and has to be SEO friendly. that's when I came across JAM stacks and found Gatsby.it was the perfect solution as it is SEO
					friendly and also integrates with netlify cms to manage the contents.
				</p>

				<p>
					The Blog is now a GitHub project and from there it directly gets uploaded to IPFS which is a peer-to-peer network. and I am using "fleek.co"
					to pin the content. As long as there are readers to read, the blog will be available, even if it gets deleted from fleek's pinning service.
					in the future the blog will have a decentralized domain name ("codacus.eth") along with the current domains ("thecodacus.com" /
					"codacus.com").
				</p>
			</PageWrapper>
		</Layout>
	)
}
